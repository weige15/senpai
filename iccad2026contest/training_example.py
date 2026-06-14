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

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

# 確保 Python 可以順利引入上一層資料夾中的官方組件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iccad2026contest.iccad2026_evaluate import (
    compute_training_loss_differentiable,
)
from lite_dataset import FloorplanDatasetLite, floorplan_collate as train_floorplan_collate

# =============================================================================
# 1. NETLIST EDGE-GNN (電路圖拓樸特徵提取器)
# =============================================================================
class NetlistGNN(nn.Module):
    def __init__(self, node_in_dim=6, hidden_dim=384):
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

    def forward(self, constraints, area_target, b2b_conn, block_count):
        N = block_count
        raw_nodes = torch.cat([constraints, area_target], dim=-1) # [N, 6]
        h = self.node_init(raw_nodes) # [N, hidden_dim]
        
        valid_mask = b2b_conn[:, 0] >= 0
        edges = b2b_conn[valid_mask]
        
        if edges.numel() > 0:
            idx_i = edges[:, 0].long()
            idx_j = edges[:, 1].long()
            weight = edges[:, 2].unsqueeze(-1)
            
            msg_to_i = h[idx_j] * weight
            msg_to_j = h[idx_i] * weight
            
            agg_msg = torch.zeros_like(h)
            agg_msg.index_add_(0, idx_i, msg_to_i)
            agg_msg.index_add_(0, idx_j, msg_to_j)
            
            h = self.msg_merge(torch.cat([h, agg_msg], dim=-1))
            
        return h


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

    def forward(self, x, c_node):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_node).chunk(6, dim=-1)
        res_attn = (self.norm1(x) * (1 + scale_msa) + shift_msa)
        attn_out, _ = self.attn(res_attn, res_attn, res_attn)
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
        self.coord_embedder = nn.Linear(4, hidden_size)      
        self.gnn_encoder = NetlistGNN(node_in_dim=5 + 1, hidden_dim=hidden_size)
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

    def forward(self, noised_positions, area_target, constraints, b2b_conn, t, block_count):
        x = self.coord_embedder(noised_positions)
        gnn_feats = self.gnn_encoder(constraints, area_target, b2b_conn, block_count)
        gnn_feats_b = gnn_feats.unsqueeze(0)
        t_feat = self.t_embedder(t)
        t_feat_b = t_feat.unsqueeze(1).expand(-1, block_count, -1)
        c_node = self.cond_fusion(torch.cat([gnn_feats_b, t_feat_b], dim=-1))
        
        for block in self.blocks:
            x = block(x, c_node)
            
        shift, scale = self.final_adaLN(t_feat).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        output = self.final_layer(x)
        return output


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
    parser.add_argument("--noise-std", type=float, default=5.0)
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


def find_latest_checkpoint(checkpoint_folder: Path, steps_per_epoch: int):
    if checkpoint_folder.exists():
        pth_files = list(checkpoint_folder.glob("*.pth"))
        if pth_files:
            latest_pth = None
            max_step_found = -1
            for pth in pth_files:
                step = checkpoint_step_from_name(pth, steps_per_epoch)
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


def save_checkpoint(model, save_path: Path):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(raw_model.state_dict(), save_path)


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


def compute_sample_loss(model, batch, sample_idx: int, device, max_blocks: int, noise_std: float):
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

    t_step = torch.randint(0, 1000, (1,), device=device)
    noise = torch.randn_like(gt_positions) * noise_std
    noised_input = gt_positions + noise

    predicted_denoise = model(
        noised_input.unsqueeze(0),
        valid_area,
        valid_constraints,
        sample_b2b,
        t_step,
        block_count,
    )
    positions = noised_input + predicted_denoise.squeeze(0)

    return compute_training_loss_differentiable(
        positions,
        sample_b2b,
        sample_p2b,
        sample_pins,
        sample_area[:block_count],
        sample_metrics,
    )


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

        log_main(rank, f"Detected Runtime Device: {device}")
        log_main(rank, f"World Size: {world_size}")
        log_main(rank, f"Batch Size Per Process: {args.batch_size}")

        dataset = build_training_dataset(args, rank)
        dataloader = build_dataloader(args, dataset, distributed, rank, world_size)
        log_main(rank, f"Loaded {len(dataset):,} samples and {len(dataloader):,} batches per process.\n")

        model = DiTSmallFloorplanBackbone(hidden_size=384, depth=12, num_heads=6).to(device)
        checkpoint_folder = Path(args.checkpoint_dir).expanduser()
        latest_pth, resume_global_step = find_latest_checkpoint(checkpoint_folder, len(dataloader))

        if latest_pth is not None:
            log_main(rank, "[Auto-Resume] Found historical weights in checkpoint directory.")
            log_main(rank, f"[Auto-Resume] Loading latest progress: '{latest_pth.name}'")
            state_dict = torch.load(latest_pth, map_location=device)
            model.load_state_dict(strip_module_prefix(state_dict), strict=True)
            log_main(rank, f"[Auto-Resume] Resuming from Global Step: {resume_global_step}\n")
        else:
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
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        for epoch in range(start_epoch, args.epochs + 1):
            log_main(rank, f"Starting Epoch {epoch}/{args.epochs}")
            log_main(rank, "-" * 50)

            if distributed and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)

            model.train()
            epoch_loss = 0.0
            processed_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                if epoch == start_epoch and batch_idx < start_batch_idx:
                    continue

                batch = [tensor.to(device, non_blocking=True) for tensor in batch]
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    losses = []
                    for sample_idx in range(batch[0].shape[0]):
                        loss = compute_sample_loss(
                            model,
                            batch,
                            sample_idx,
                            device,
                            args.max_blocks,
                            args.noise_std,
                        )
                        if loss is not None:
                            losses.append(loss)

                    if not losses:
                        continue

                    batch_loss = torch.stack(losses).mean()

                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                global_step += 1
                processed_batches += 1

                log_loss = reduce_mean(batch_loss, distributed, world_size)
                epoch_loss += log_loss.item()

                if args.log_freq > 0 and global_step % args.log_freq == 0:
                    log_main(
                        rank,
                        f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                        f"(Global Step {global_step}) -> Loss: {log_loss.item():.4f}",
                    )

                if args.save_freq_batches > 0 and global_step % args.save_freq_batches == 0:
                    if is_main_process(rank):
                        checkpoint_folder.mkdir(exist_ok=True, parents=True)
                        checkpoint_name = f"dit_gnn_step_{global_step}_loss_{log_loss.item():.2f}.pth"
                        save_path = checkpoint_folder / checkpoint_name
                        save_checkpoint(model, save_path)
                        print(f"    [Checkpoint] Saved: {save_path}", flush=True)

            if processed_batches > 0:
                avg_epoch_loss = epoch_loss / processed_batches
                log_main(rank, f"Epoch {epoch} Completed. Average Loss: {avg_epoch_loss:.4f}")

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
