#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Edge-GNN + DiT-Small Training Pipeline

功能升級（針對 14 天長週期訓練優化）：
  1. 10-Batch 高頻存檔：由原本的 Epoch 存檔改為每 10 個 batches (steps) 就立刻儲存一次 
     checkpoint，徹底避免因系統中斷、超時導致的算力浪費。
  2. 智慧續訓 (Step 級 Auto-Resume)：啟動時自動掃描 checkpoints/，不論是 epoch 命名
     還是 step 命名，皆能精準提取目前跑得最遠（步數最大）的權重直接載入，並同步將進度推至下一空步。
  3. 參數鎖定：維持原本測試最穩定的學習率、加噪強度與優化器配置，不作任何變動。
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path
import math
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

# 確保 Python 可以順利引入上一層資料夾中的官方組件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iccad2026contest.iccad2026_evaluate import (
    compute_training_loss_differentiable,
    compute_training_loss_differentiable_batch,
)
from lite_dataset import FloorplanDatasetLite, floorplan_collate as train_floorplan_collate

# =============================================================================
# 1. NETLIST EDGE-GNN (電路圖拓樸特徵提取器)
# =============================================================================
PIN_FEATURE_DIM = 3


class NetlistGNN(nn.Module):
    def __init__(self, node_in_dim=6 + PIN_FEATURE_DIM, hidden_dim=384):
        super().__init__()
        self.node_init = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.msg_merge = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, constraints, area_target, b2b_conn, p2b_conn, pins_pos, block_count):
        single_sample = constraints.dim() == 2
        if single_sample:
            constraints = constraints.unsqueeze(0)
            area_target = area_target.unsqueeze(0)
            b2b_conn = b2b_conn.unsqueeze(0)
            if p2b_conn is not None:
                p2b_conn = p2b_conn.unsqueeze(0)
            if pins_pos is not None:
                pins_pos = pins_pos.unsqueeze(0)

        if area_target.dim() == 2:
            area_target = area_target.unsqueeze(-1)

        batch_size, n_blocks, _ = constraints.shape
        device = constraints.device
        if isinstance(block_count, torch.Tensor):
            if block_count.dtype == torch.bool and block_count.dim() == 2:
                node_mask = block_count.to(device=device)
            else:
                counts = block_count.to(device=device).long().view(-1)
                node_mask = torch.arange(n_blocks, device=device).unsqueeze(0) < counts.unsqueeze(1)
        elif block_count is None:
            node_mask = area_target.squeeze(-1) > 0
        else:
            counts = torch.full((batch_size,), int(block_count), device=device, dtype=torch.long)
            node_mask = torch.arange(n_blocks, device=device).unsqueeze(0) < counts.unsqueeze(1)

        pin_features = self._pin_features(area_target, p2b_conn, pins_pos, node_mask)
        raw_nodes = torch.cat([constraints, area_target, pin_features], dim=-1)
        h = self.node_init(raw_nodes)
        h = h * node_mask.unsqueeze(-1).to(dtype=h.dtype)

        if b2b_conn.numel() > 0:
            idx_i_raw = b2b_conn[:, :, 0]
            idx_j_raw = b2b_conn[:, :, 1]
            idx_i = idx_i_raw.clamp(min=0, max=max(n_blocks - 1, 0)).long()
            idx_j = idx_j_raw.clamp(min=0, max=max(n_blocks - 1, 0)).long()
            valid_edges = (
                (idx_i_raw >= 0) & (idx_j_raw >= 0)
                & (idx_i_raw < n_blocks) & (idx_j_raw < n_blocks)
                & node_mask.gather(1, idx_i)
                & node_mask.gather(1, idx_j)
            )

            idx_i_expanded = idx_i.unsqueeze(-1).expand(-1, -1, h.size(-1))
            idx_j_expanded = idx_j.unsqueeze(-1).expand(-1, -1, h.size(-1))
            h_i = h.gather(1, idx_i_expanded)
            h_j = h.gather(1, idx_j_expanded)
            weight = (
                b2b_conn[:, :, 2].to(dtype=h.dtype).unsqueeze(-1)
                * valid_edges.unsqueeze(-1).to(dtype=h.dtype)
            )

            agg_msg = torch.zeros_like(h)
            agg_msg.scatter_add_(1, idx_i_expanded, h_j * weight)
            agg_msg.scatter_add_(1, idx_j_expanded, h_i * weight)

            h = self.msg_merge(torch.cat([h, agg_msg], dim=-1))
            h = h * node_mask.unsqueeze(-1).to(dtype=h.dtype)

        return h.squeeze(0) if single_sample else h

    @staticmethod
    def _pin_features(area_target, p2b_conn, pins_pos, node_mask):
        batch_size, n_blocks = node_mask.shape
        features = torch.zeros(
            batch_size,
            n_blocks,
            PIN_FEATURE_DIM,
            device=area_target.device,
            dtype=area_target.dtype,
        )
        if p2b_conn is None or pins_pos is None or p2b_conn.numel() == 0 or pins_pos.numel() == 0:
            return features
        if p2b_conn.dim() == 2:
            p2b_conn = p2b_conn.unsqueeze(0)
        if pins_pos.dim() == 2:
            pins_pos = pins_pos.unsqueeze(0)

        n_pins = pins_pos.shape[1]
        pin_raw = p2b_conn[:, :, 0]
        block_raw = p2b_conn[:, :, 1]
        pin_idx = pin_raw.clamp(min=0, max=max(n_pins - 1, 0)).long()
        block_idx = block_raw.clamp(min=0, max=max(n_blocks - 1, 0)).long()
        valid = (
            (pin_raw >= 0) & (block_raw >= 0)
            & (pin_raw < n_pins) & (block_raw < n_blocks)
            & node_mask.gather(1, block_idx)
        )

        pin_x = pins_pos[:, :, 0].to(dtype=area_target.dtype).gather(1, pin_idx)
        pin_y = pins_pos[:, :, 1].to(dtype=area_target.dtype).gather(1, pin_idx)
        valid = valid & torch.isfinite(pin_x) & torch.isfinite(pin_y)
        weight = p2b_conn[:, :, 2].to(dtype=area_target.dtype).clamp_min(0.0)
        weight = weight * valid.to(dtype=area_target.dtype)

        scale = torch.sqrt(
            (area_target.squeeze(-1).clamp_min(0.0) * node_mask.to(dtype=area_target.dtype))
            .sum(dim=1)
            .clamp_min(1.0)
        ).view(batch_size, 1)
        contributions = torch.stack(
            (
                weight,
                weight * pin_x / scale,
                weight * pin_y / scale,
            ),
            dim=-1,
        )
        features.scatter_add_(
            1,
            block_idx.unsqueeze(-1).expand(-1, -1, PIN_FEATURE_DIM),
            contributions,
        )
        weight_sum = features[:, :, 0:1]
        features[:, :, 1:3] = features[:, :, 1:3] / weight_sum.clamp_min(1e-6)
        features[:, :, 0:1] = torch.log1p(weight_sum)
        return features * node_mask.unsqueeze(-1).to(dtype=features.dtype)


# =============================================================================
# 2. TIME EMBEDDING & ADALN MODULATOR
# =============================================================================
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# =============================================================================
# 3. DiT-SMALL TRANSFORMER BLOCK WITH STRONG ADALN
# =============================================================================
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c_node, key_padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_node).chunk(6, dim=-1)
        res_attn = (self.norm1(x) * (1 + scale_msa) + shift_msa)
        attn_out, _ = self.attn(
            res_attn,
            res_attn,
            res_attn,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_msa * attn_out
        res_mlp = (self.norm2(x) * (1 + scale_mlp) + shift_mlp)
        x = x + gate_mlp * self.mlp(res_mlp)
        return x


# =============================================================================
# 4. UPDATED DiT-SMALL BACKBONE
# =============================================================================
class DiTSmallFloorplanBackbone(nn.Module):
    def __init__(self, hidden_size=384, depth=12, num_heads=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("coordinate_contract_version", torch.tensor(2, dtype=torch.long))
        self.coord_embedder = nn.Linear(4, hidden_size)      
        self.gnn_encoder = NetlistGNN(node_in_dim=5 + 1 + PIN_FEATURE_DIM, hidden_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, 4)
        )
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, noised_positions, area_target, constraints, b2b_conn, p2b_conn, pins_pos, t, block_count):
        if noised_positions.dim() == 2:
            noised_positions = noised_positions.unsqueeze(0)
        batch_size, n_blocks, _ = noised_positions.shape

        if area_target.dim() == 1:
            area_target = area_target.unsqueeze(0).unsqueeze(-1)
        elif area_target.dim() == 2:
            if area_target.shape == noised_positions.shape[:2]:
                area_target = area_target.unsqueeze(-1)
            else:
                area_target = area_target.unsqueeze(0)

        if constraints.dim() == 2:
            constraints = constraints.unsqueeze(0)
        if b2b_conn.dim() == 2:
            b2b_conn = b2b_conn.unsqueeze(0)
        if p2b_conn is not None and p2b_conn.dim() == 2:
            p2b_conn = p2b_conn.unsqueeze(0)
        if pins_pos is not None and pins_pos.dim() == 2:
            pins_pos = pins_pos.unsqueeze(0)

        if isinstance(block_count, torch.Tensor):
            if block_count.dtype == torch.bool and block_count.dim() == 2:
                block_mask = block_count.to(device=noised_positions.device)
            else:
                counts = block_count.to(device=noised_positions.device).long().view(-1)
                block_mask = torch.arange(n_blocks, device=noised_positions.device).unsqueeze(0) < counts.unsqueeze(1)
        elif block_count is None:
            block_mask = area_target.squeeze(-1) > 0
        else:
            counts = torch.full((batch_size,), int(block_count), device=noised_positions.device, dtype=torch.long)
            block_mask = torch.arange(n_blocks, device=noised_positions.device).unsqueeze(0) < counts.unsqueeze(1)

        if t.dim() == 0:
            t = t.view(1)
        if t.numel() == 1 and batch_size > 1:
            t = t.expand(batch_size)

        coord_scale = torch.sqrt(
            (area_target.squeeze(-1).clamp_min(0.0) * block_mask.to(dtype=area_target.dtype))
            .sum(dim=1)
            .clamp_min(1.0)
        ).view(batch_size, 1, 1).to(device=noised_positions.device, dtype=noised_positions.dtype)

        x = self.coord_embedder(noised_positions / coord_scale)
        x = x * block_mask.unsqueeze(-1).to(dtype=x.dtype)
        gnn_feats = self.gnn_encoder(constraints, area_target, b2b_conn, p2b_conn, pins_pos, block_count)
        t_feat = self.t_embedder(t)
        t_feat_b = t_feat.unsqueeze(1).expand(-1, n_blocks, -1)
        c_node = self.cond_fusion(torch.cat([gnn_feats, t_feat_b], dim=-1))
        key_padding_mask = ~block_mask

        for block in self.blocks:
            x = block(x, c_node, key_padding_mask=key_padding_mask)
            x = x * block_mask.unsqueeze(-1).to(dtype=x.dtype)

        shift, scale = self.final_adaLN(t_feat).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        output = self.final_layer(x) * coord_scale
        return output * block_mask.unsqueeze(-1).to(dtype=output.dtype)


# =============================================================================
# 5. DATA, DISTRIBUTED, AND TRAINING HELPERS
# =============================================================================

class FloorplanWorkerDataset(Dataset):
    """FloorSet-Lite dataset reader for an existing worker_* directory tree."""

    layouts_per_file = 112

    def __init__(self, worker_root: Path):
        self.worker_root = Path(worker_root)
        self.all_files = []
        for worker_idx in range(100):
            self.all_files.extend(sorted(glob.glob(str(
                self.worker_root / f"worker_{worker_idx}" / "layouts*"
            ))))

        if not self.all_files:
            self.all_files = sorted(glob.glob(str(self.worker_root / "worker_*" / "layouts*")))

        if not self.all_files:
            raise FileNotFoundError(
                f"No layouts* files found under worker directories in {self.worker_root}"
            )

        self.cached_file_idx = -1
        self.cached_input_file_contents = None

    def __len__(self):
        return len(self.all_files) * self.layouts_per_file

    def __getitem__(self, idx):
        file_idx, layout_idx = divmod(idx, self.layouts_per_file)
        if file_idx != self.cached_file_idx:
            self.cached_input_file_contents = torch.load(self.all_files[file_idx], map_location="cpu")
            self.cached_file_idx = file_idx

        area_target = self.cached_input_file_contents[0][layout_idx][:, 0]
        placement_constraints = self.cached_input_file_contents[0][layout_idx][:, 1:]
        b2b_connectivity = self.cached_input_file_contents[1][layout_idx]
        p2b_connectivity = self.cached_input_file_contents[2][layout_idx]
        pins_pos = self.cached_input_file_contents[3][layout_idx]

        tree_sol = self.cached_input_file_contents[4][layout_idx]
        fp_sol = self.cached_input_file_contents[5][layout_idx]
        metrics_sol = self.cached_input_file_contents[6][layout_idx]

        return {
            "input": (
                area_target,
                b2b_connectivity,
                p2b_connectivity,
                pins_pos,
                placement_constraints,
            ),
            "label": (tree_sol, fp_sol, metrics_sol),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Edge-GNN + DiT-Small FloorSet model."
    )
    parser.add_argument(
        "--data-path",
        default="../",
        help=(
            "Dataset root. Accepts a FloorSet root containing floorset_lite/worker_*, "
            "a project root containing data_lite/worker_*, or a direct worker_* root."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1, help="Samples per optimizer step per process.")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit samples for smoke tests.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Legacy coordinate jitter. Keep 0.0 for rectified one-step training that matches inference.",
    )
    parser.add_argument(
        "--supervised-weight",
        type=float,
        default=1.0,
        help="Weight for normalized SmoothL1 x/y supervision against training labels.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm before optimizer step. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-loss",
        type=float,
        default=100.0,
        help="Skip optimizer step when finite loss exceeds this value. Use 0 to disable.",
    )
    parser.add_argument("--max-blocks", type=int, default=1000)
    parser.add_argument("--save-freq-batches", type=int, default=10)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(Path(__file__).parent / "checkpoints"),
        help="Directory for auto-resume and checkpoint writes.",
    )
    parser.add_argument("--device", type=int, default=None, help="Single-process CUDA device index.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision.")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Fall back to the official loader, which may prompt/download if data is missing.",
    )
    parser.add_argument("--drop-last", action="store_true")
    parser.set_defaults(shuffle=True, pin_memory=True)
    parser.add_argument("--shuffle", action="store_true", dest="shuffle")
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    return parser.parse_args()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)

    return distributed, rank, local_rank, world_size


def cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def log_main(rank: int, message: str):
    if is_main_process(rank):
        print(message, flush=True)


def find_worker_root(data_path: str):
    root = Path(data_path).expanduser().resolve()
    candidates = [
        root / "floorset_lite",
        root / "data_lite",
        root,
    ]
    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("worker_*")):
            return candidate
    return None


def build_training_dataset(args, rank: int):
    worker_root = find_worker_root(args.data_path)
    if worker_root is not None:
        log_main(rank, f"Using existing training data: {worker_root}")
        dataset = FloorplanWorkerDataset(worker_root)
    elif args.allow_download:
        log_main(rank, f"No existing worker_* tree found. Falling back to official loader at {args.data_path}.")
        dataset = FloorplanDatasetLite(args.data_path)
    else:
        root = Path(args.data_path).expanduser().resolve()
        raise FileNotFoundError(
            "Training data not found. Expected one of:\n"
            f"  {root / 'floorset_lite'}/worker_*\n"
            f"  {root / 'data_lite'}/worker_*\n"
            f"  {root}/worker_*\n"
            "Pass --data-path to the correct root, or use --allow-download explicitly."
        )

    if args.num_samples is not None:
        dataset = Subset(dataset, list(range(min(args.num_samples, len(dataset)))))

    return dataset


def build_dataloader(args, dataset, distributed: bool, rank: int, world_size: int):
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=args.shuffle,
            drop_last=args.drop_last,
        )

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(args.shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and torch.cuda.is_available()),
        drop_last=args.drop_last,
        collate_fn=train_floorplan_collate,
        **loader_kwargs,
    )


def checkpoint_step_from_name(path: Path, steps_per_epoch: int) -> int:
    step_match = re.search(r"step_(\d+)", path.name)
    if step_match:
        return int(step_match.group(1))

    epoch_match = re.search(r"epoch_(\d+)", path.name)
    if epoch_match:
        return int(epoch_match.group(1)) * steps_per_epoch

    return -1


def checkpoint_loss_is_finite(path: Path) -> bool:
    loss_match = re.search(r"loss_([^_]+)", path.stem)
    if not loss_match:
        return True
    try:
        return math.isfinite(float(loss_match.group(1)))
    except ValueError:
        return True


def find_latest_checkpoint(checkpoint_folder: Path, steps_per_epoch: int, before_step=None):
    if checkpoint_folder.exists():
        pth_files = list(checkpoint_folder.glob("*.pth"))
        if pth_files:
            latest_pth = None
            max_step_found = -1
            for pth in pth_files:
                if not checkpoint_loss_is_finite(pth):
                    continue
                step = checkpoint_step_from_name(pth, steps_per_epoch)
                if before_step is not None and step >= before_step:
                    continue
                if step > max_step_found:
                    max_step_found = step
                    latest_pth = pth
            if latest_pth is not None and max_step_found >= 0:
                return latest_pth, max_step_found
    return None, 0


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def state_dict_is_finite(state_dict) -> bool:
    for value in state_dict.values():
        if (
            torch.is_tensor(value)
            and (value.is_floating_point() or value.is_complex())
            and not torch.isfinite(value).all().item()
        ):
            return False
    return True


def raw_model_for_checkpoint(model):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return raw_model


def model_state_is_finite(model) -> bool:
    return state_dict_is_finite(raw_model_for_checkpoint(model).state_dict())


def model_gradients_are_finite(model) -> bool:
    for param in raw_model_for_checkpoint(model).parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all().item():
            return False
    return True


def save_checkpoint(model, save_path: Path):
    state_dict = raw_model_for_checkpoint(model).state_dict()
    if not state_dict_is_finite(state_dict):
        raise FloatingPointError(f"Refusing to save non-finite checkpoint: {save_path}")
    torch.save(state_dict, save_path)


def set_scheduler_step(scheduler, optimizer, lr_lambda, global_step: int):
    scheduler.last_epoch = global_step
    scale = lr_lambda(global_step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale


def reduce_mean(value: torch.Tensor, distributed: bool, world_size: int):
    if distributed:
        value = value.detach().clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
        return value
    return value.detach()


def all_ranks_true(local_value: bool, device, distributed: bool) -> bool:
    flag = torch.tensor(1 if local_value else 0, device=device, dtype=torch.int32)
    if distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def loss_is_acceptable(loss: torch.Tensor, max_loss: float) -> bool:
    if not torch.isfinite(loss).all().item():
        return False
    if max_loss > 0 and abs(float(loss.detach().item())) > max_loss:
        return False
    return True


def _finite_float(value, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(value):
            value = value.detach().item()
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def build_initial_layout_batch(area_target: torch.Tensor, constraints: torch.Tensor, fp_sol: torch.Tensor) -> torch.Tensor:
    batch_size, n_blocks = area_target.shape
    result = torch.zeros(batch_size, n_blocks, 4, device=area_target.device, dtype=fp_sol.dtype)
    valid_mask = area_target > 0
    constraint_cols = constraints.shape[2] if constraints.dim() == 3 else 0

    for sample_idx in range(batch_size):
        dims: List[Tuple[float, float]] = []
        valid_indices = []
        for block_idx in range(n_blocks):
            if not bool(valid_mask[sample_idx, block_idx].item()):
                dims.append((0.0, 0.0))
                continue
            area = max(_finite_float(area_target[sample_idx, block_idx], 1.0), 1e-6)
            side = math.sqrt(area)
            is_fixed = constraint_cols > 0 and _finite_float(constraints[sample_idx, block_idx, 0]) != 0.0
            is_preplaced = constraint_cols > 1 and _finite_float(constraints[sample_idx, block_idx, 1]) != 0.0
            target_w = _finite_float(fp_sol[sample_idx, block_idx, 0], side)
            target_h = _finite_float(fp_sol[sample_idx, block_idx, 1], side)
            if (is_fixed or is_preplaced) and target_w > 0.0 and target_h > 0.0:
                width, height = target_w, target_h
            else:
                width, height = side, side
            dims.append((width, height))
            valid_indices.append(block_idx)

        total_area = sum(width * height for width, height in dims)
        target_row_width = max(
            1.0,
            math.sqrt(max(total_area, 1.0)),
            max((dims[i][0] for i in valid_indices), default=1.0),
        )
        x = 0.0
        y = 0.0
        row_height = 0.0
        for block_idx in valid_indices:
            width, height = dims[block_idx]
            is_preplaced = constraint_cols > 1 and _finite_float(constraints[sample_idx, block_idx, 1]) != 0.0
            if is_preplaced:
                px = _finite_float(fp_sol[sample_idx, block_idx, 2], float("nan"))
                py = _finite_float(fp_sol[sample_idx, block_idx, 3], float("nan"))
                if math.isfinite(px) and math.isfinite(py):
                    result[sample_idx, block_idx] = torch.tensor(
                        (px, py, width, height),
                        device=area_target.device,
                        dtype=fp_sol.dtype,
                    )
                    continue
            if x > 0.0 and x + width > target_row_width:
                x = 0.0
                y += row_height
                row_height = 0.0
            result[sample_idx, block_idx] = torch.tensor(
                (x, y, width, height),
                device=area_target.device,
                dtype=fp_sol.dtype,
            )
            x += width
            row_height = max(row_height, height)

    return result


def supervised_xy_loss(positions: torch.Tensor, ground_truth: torch.Tensor, area_target: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
    scale = torch.sqrt(
        (area_target.clamp_min(0.0) * block_mask.to(dtype=positions.dtype))
        .sum(dim=1)
        .clamp_min(1.0)
    ).view(-1, 1, 1)
    normalized_error = (positions[:, :, :2] - ground_truth[:, :, :2]) / scale
    per_block = F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        reduction="none",
    ).sum(dim=-1)
    return (
        per_block * block_mask.to(dtype=positions.dtype)
    ).sum(dim=1) / block_mask.sum(dim=1).to(dtype=positions.dtype).clamp_min(1.0)


def compute_sample_loss(model, batch, sample_idx: int, device, max_blocks: int, noise_std: float, supervised_weight: float):
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, _tree_sol, fp_sol, metrics = batch

    sample_area = area_target[sample_idx]
    block_count = int((sample_area != -1).sum().item())
    if block_count <= 0:
        return None
    block_count = min(block_count, max_blocks)

    sample_b2b = b2b_conn[sample_idx]
    sample_p2b = p2b_conn[sample_idx]
    sample_pins = pins_pos[sample_idx]
    sample_constraints = constraints[sample_idx]
    sample_metrics = metrics[sample_idx]
    sample_fp = fp_sol[sample_idx]

    valid_area = sample_area[:block_count].unsqueeze(-1)
    valid_constraints = sample_constraints[:block_count].float()
    ground_truth = sample_fp[:block_count]

    gt_positions = torch.stack([
        ground_truth[:, 2],
        ground_truth[:, 3],
        ground_truth[:, 0],
        ground_truth[:, 1],
    ], dim=1)

    base_layout = build_initial_layout_batch(
        sample_area[:block_count].unsqueeze(0),
        valid_constraints.unsqueeze(0),
        sample_fp[:block_count].unsqueeze(0),
    ).squeeze(0)
    if noise_std > 0:
        base_layout[:, :2] = base_layout[:, :2] + torch.randn_like(base_layout[:, :2]) * noise_std
    t_step = torch.zeros((1,), device=device, dtype=torch.long)

    predicted_delta = model(
        base_layout.unsqueeze(0),
        valid_area,
        valid_constraints,
        sample_b2b,
        sample_p2b,
        sample_pins,
        t_step,
        block_count,
    )
    positions = base_layout.clone()
    positions[:, :2] = positions[:, :2] + predicted_delta.squeeze(0)[:, :2]

    proxy_loss = compute_training_loss_differentiable(
        positions,
        sample_b2b,
        sample_p2b,
        sample_pins,
        sample_area[:block_count],
        sample_metrics,
    )
    xy_loss = supervised_xy_loss(
        positions.unsqueeze(0),
        gt_positions.unsqueeze(0),
        sample_area[:block_count].unsqueeze(0),
        torch.ones((1, block_count), device=device, dtype=torch.bool),
    )[0]
    return proxy_loss + supervised_weight * xy_loss


def compute_batch_loss(
    model,
    batch,
    device,
    max_blocks: int,
    noise_std: float,
    supervised_weight: float,
    return_components: bool = False,
):
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, _tree_sol, fp_sol, metrics = batch

    n_blocks = area_target.shape[1]
    if max_blocks is not None:
        n_blocks = min(n_blocks, max_blocks)

    area_target = area_target[:, :n_blocks]
    constraints = constraints[:, :n_blocks].float()
    fp_sol = fp_sol[:, :n_blocks].float()
    block_mask = area_target > 0
    valid_samples = block_mask.any(dim=1)
    if not valid_samples.any():
        return None

    ground_truth = torch.stack([
        fp_sol[:, :, 2],
        fp_sol[:, :, 3],
        fp_sol[:, :, 0],
        fp_sol[:, :, 1],
    ], dim=-1)
    ground_truth = ground_truth.masked_fill(~block_mask.unsqueeze(-1), 0.0)

    base_layout = build_initial_layout_batch(area_target, constraints, fp_sol)
    if noise_std > 0:
        jitter = torch.randn_like(base_layout[:, :, :2]) * noise_std
        base_layout[:, :, :2] = base_layout[:, :, :2] + jitter * block_mask.unsqueeze(-1).to(dtype=jitter.dtype)
    t_step = torch.zeros((area_target.shape[0],), device=device, dtype=torch.long)

    predicted_delta = model(
        base_layout,
        area_target.unsqueeze(-1).float(),
        constraints,
        b2b_conn,
        p2b_conn,
        pins_pos,
        t_step,
        block_mask,
    )
    positions = base_layout.clone()
    positions[:, :, :2] = positions[:, :, :2] + predicted_delta[:, :, :2]

    proxy_losses = compute_training_loss_differentiable_batch(
        positions,
        b2b_conn,
        p2b_conn,
        pins_pos,
        area_target,
        metrics,
        block_mask=block_mask,
    )
    xy_losses = supervised_xy_loss(positions, ground_truth, area_target, block_mask)
    losses = proxy_losses + supervised_weight * xy_losses
    valid_losses = losses[valid_samples]
    mean_loss = valid_losses.mean()
    if return_components:
        return (
            mean_loss,
            proxy_losses[valid_samples].mean().detach(),
            xy_losses[valid_samples].mean().detach(),
        )
    return mean_loss


# =============================================================================
# 6. MAIN TRAINING LOOP (WITH BATCHING AND DDP)
# =============================================================================
def main():
    args = parse_args()
    distributed, rank, local_rank, world_size = setup_distributed()

    try:
        log_main(rank, "=" * 70)
        log_main(rank, "ICCAD 2026 FloorSet Challenge - Batched/DDP Training Pipeline")
        log_main(rank, "=" * 70)

        if distributed:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        elif torch.cuda.is_available():
            if args.device is not None:
                torch.cuda.set_device(args.device)
                device = torch.device(f"cuda:{args.device}")
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

        log_main(rank, f"Detected Runtime Device: {device}")
        log_main(rank, f"World Size: {world_size}")
        log_main(rank, f"Batch Size Per Process: {args.batch_size}")

        dataset = build_training_dataset(args, rank)
        dataloader = build_dataloader(args, dataset, distributed, rank, world_size)
        log_main(rank, f"Loaded {len(dataset):,} samples and {len(dataloader):,} batches per process.\n")

        model = DiTSmallFloorplanBackbone(hidden_size=384, depth=12, num_heads=6).to(device)
        checkpoint_folder = Path(args.checkpoint_dir).expanduser()
        latest_pth, resume_global_step = find_latest_checkpoint(checkpoint_folder, len(dataloader))

        loaded_checkpoint = False
        while latest_pth is not None:
            log_main(rank, "[Auto-Resume] Found historical weights in checkpoint directory.")
            log_main(rank, f"[Auto-Resume] Checking checkpoint: '{latest_pth.name}'")
            try:
                state_dict = strip_module_prefix(torch.load(latest_pth, map_location=device))
                if state_dict_is_finite(state_dict):
                    model.load_state_dict(state_dict, strict=True)
                    loaded_checkpoint = True
                    log_main(rank, f"[Auto-Resume] Loaded checkpoint: '{latest_pth.name}'")
                    log_main(rank, f"[Auto-Resume] Resuming from Global Step: {resume_global_step}\n")
                    break
                log_main(rank, f"[Auto-Resume] Skipping non-finite checkpoint: '{latest_pth.name}'")
            except Exception as exc:
                log_main(rank, f"[Auto-Resume] Skipping incompatible checkpoint '{latest_pth.name}': {exc}")
            latest_pth, resume_global_step = find_latest_checkpoint(
                checkpoint_folder,
                len(dataloader),
                before_step=resume_global_step,
            )

        if not loaded_checkpoint:
            resume_global_step = 0
            log_main(rank, "[Auto-Resume] No usable checkpoint found. Starting from scratch.")

        if distributed:
            if device.type == "cuda":
                model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            else:
                model = DistributedDataParallel(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        total_steps = args.epochs * len(dataloader)
        if total_steps <= 0:
            raise RuntimeError("Training dataloader is empty.")

        warmup_steps = int(total_steps * 0.1)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if resume_global_step > 0:
            set_scheduler_step(scheduler, optimizer, lr_lambda, resume_global_step)
            log_main(rank, f"[Auto-Resume] LR scheduler synchronized to step {resume_global_step}.")

        start_epoch = (resume_global_step // len(dataloader)) + 1
        start_batch_idx = resume_global_step % len(dataloader)
        global_step = resume_global_step
        amp_enabled = args.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

        for epoch in range(start_epoch, args.epochs + 1):
            log_main(rank, f"Starting Epoch {epoch}/{args.epochs}")
            log_main(rank, "-" * 50)

            if distributed and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)

            model.train()
            epoch_loss = torch.zeros((), device=device)
            epoch_proxy_loss = torch.zeros((), device=device)
            epoch_xy_loss = torch.zeros((), device=device)
            processed_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                if epoch == start_epoch and batch_idx < start_batch_idx:
                    continue

                batch = [tensor.to(device, non_blocking=True) for tensor in batch]
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    loss_result = compute_batch_loss(
                        model,
                        batch,
                        device,
                        args.max_blocks,
                        args.noise_std,
                        args.supervised_weight,
                        return_components=True,
                    )
                    if loss_result is None:
                        continue
                    batch_loss, batch_proxy_loss, batch_xy_loss = loss_result

                local_loss_ok = loss_is_acceptable(batch_loss, args.max_loss)
                if not all_ranks_true(local_loss_ok, device, distributed):
                    loss_value = (
                        float(batch_loss.detach().item())
                        if torch.isfinite(batch_loss).all().item()
                        else float("nan")
                    )
                    log_main(
                        rank,
                        f"  Batch [{batch_idx + 1}/{len(dataloader)}] skipped: "
                        f"non-finite or runaway loss ({loss_value:.4f})",
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(batch_loss).backward()
                grad_norm = None
                gradients_unscaled = False
                if args.grad_clip > 0:
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                        gradients_unscaled = True
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                local_grad_ok = True
                if grad_norm is not None:
                    local_grad_ok = torch.isfinite(grad_norm).all().item()
                else:
                    local_grad_ok = model_gradients_are_finite(model)
                if not all_ranks_true(local_grad_ok, device, distributed):
                    log_main(
                        rank,
                        f"  Batch [{batch_idx + 1}/{len(dataloader)}] skipped: non-finite gradients",
                    )
                    if amp_enabled and gradients_unscaled:
                        # Reset GradScaler state after unscale_ even when this batch is skipped.
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scale_before_step = scaler.get_scale() if amp_enabled else None
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_skipped = (
                    amp_enabled and scaler.get_scale() < scale_before_step
                )
                if not optimizer_step_skipped:
                    scheduler.step()

                global_step += 1
                processed_batches += 1

                log_loss = reduce_mean(batch_loss, distributed, world_size)
                log_proxy_loss = reduce_mean(batch_proxy_loss, distributed, world_size)
                log_xy_loss = reduce_mean(batch_xy_loss, distributed, world_size)
                epoch_loss = epoch_loss + log_loss
                epoch_proxy_loss = epoch_proxy_loss + log_proxy_loss
                epoch_xy_loss = epoch_xy_loss + log_xy_loss

                if args.log_freq > 0 and global_step % args.log_freq == 0:
                    log_loss_value = log_loss.item()
                    log_proxy_value = log_proxy_loss.item()
                    log_xy_value = log_xy_loss.item()
                    log_main(
                        rank,
                        f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                        f"(Global Step {global_step}) -> Loss: {log_loss_value:.4f} "
                        f"(proxy={log_proxy_value:.4f}, xy={log_xy_value:.4f}, "
                        f"wxy={args.supervised_weight * log_xy_value:.4f})",
                    )

                if args.save_freq_batches > 0 and global_step % args.save_freq_batches == 0:
                    if not all_ranks_true(model_state_is_finite(model), device, distributed):
                        raise FloatingPointError(
                            "Model parameters became non-finite; restart from the last finite checkpoint."
                        )
                    if is_main_process(rank):
                        log_loss_value = log_loss.item()
                        checkpoint_folder.mkdir(exist_ok=True, parents=True)
                        checkpoint_name = f"dit_gnn_step_{global_step}_loss_{log_loss_value:.2f}.pth"
                        save_path = checkpoint_folder / checkpoint_name
                        save_checkpoint(model, save_path)
                        print(f"    [Checkpoint] Saved: {save_path}", flush=True)

            if processed_batches > 0:
                avg_epoch_loss = (epoch_loss / processed_batches).item()
                avg_epoch_proxy = (epoch_proxy_loss / processed_batches).item()
                avg_epoch_xy = (epoch_xy_loss / processed_batches).item()
                log_main(
                    rank,
                    f"Epoch {epoch} Completed. Average Loss: {avg_epoch_loss:.4f} "
                    f"(proxy={avg_epoch_proxy:.4f}, xy={avg_epoch_xy:.4f}, "
                    f"wxy={args.supervised_weight * avg_epoch_xy:.4f})",
                )

            if not all_ranks_true(model_state_is_finite(model), device, distributed):
                raise FloatingPointError(
                    "Model parameters became non-finite; restart from the last finite checkpoint."
                )

            if is_main_process(rank):
                checkpoint_folder.mkdir(exist_ok=True, parents=True)
                epoch_save_name = f"dit_gnn_step_{global_step}_epoch_{epoch}_end.pth"
                save_checkpoint(model, checkpoint_folder / epoch_save_name)

        log_main(rank, "=" * 70)
        log_main(rank, "Edge-GNN + DiT-Small training completed.")
        log_main(rank, "=" * 70)
    finally:
        cleanup_distributed(distributed)


if __name__ == '__main__':
    main()
