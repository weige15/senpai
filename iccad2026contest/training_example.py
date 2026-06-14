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

import sys
import os
import re
from pathlib import Path
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 確保 Python 可以順利引入上一層資料夾中的官方組件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iccad2026contest.iccad2026_evaluate import (
    get_training_dataloader,
    compute_training_loss_differentiable,
)

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
# 5. MAIN TRAINING LOOP (WITH HIGH-FREQUENCY CHECKPOINTING)
# =============================================================================
def main():
    print("="*70)
    print("ICCAD 2026 FloorSet Challenge - High-Frequency Checkpoint Training Pipeline")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected Runtime Device: {device}")
    
    NUM_EPOCHS = 100  
    MAX_BLOCKS_LIMIT = 1000 
    SAVE_FREQ_BATCHES = 10  # 🚀 大改進：每 10 個 batches 就自動儲存一次權重
    
    model = DiTSmallFloorplanBackbone(hidden_size=384, depth=12, num_heads=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # 💥 依照指示維持原本的 LR

    print("\nLoading training data loader...")
    dataloader = get_training_dataloader(
        batch_size=1,
        num_samples=None,  # 開啟全量 1M 大規模訓練
        shuffle=True    
    )
    print(f"Successfully loaded {len(dataloader)} samples.\n")

    # ─── 🔍 智慧續訓擴充：能同時解讀 Epoch 或是 Step 命名的歷史文件 ───
    checkpoint_folder = Path(__file__).parent / "checkpoints"
    resume_global_step = 0  # 紀錄全域總共已經跑了多少步
    
    if checkpoint_folder.exists():
        pth_files = list(checkpoint_folder.glob("*.pth"))
        if pth_files:
            latest_pth = None
            max_step_found = -1
            
            for pth in pth_files:
                # 優先匹配新版的 step 命名格式
                step_match = re.search(r"step_(\d+)", pth.name)
                if step_match:
                    st_num = int(step_match.group(1))
                    if st_num > max_step_found:
                        max_step_found = st_num
                        latest_pth = pth
                else:
                    # 兼容舊版的 epoch 命名格式，換算成步數
                    epoch_match = re.search(r"epoch_(\d+)", pth.name)
                    if epoch_match:
                        ep_num = int(epoch_match.group(1))
                        st_num = ep_num * len(dataloader) # 舊版一輪等於一個全長 dataloader 步數
                        if st_num > max_step_found:
                            max_step_found = st_num
                            latest_pth = pth
            
            if latest_pth and max_step_found != -1:
                print(f"[Auto-Resume] Found historical weights in checkpoint directory.")
                print(f"[Auto-Resume] Loading latest progress: '{latest_pth.name}'")
                
                # 載入歷史模型
                state_dict = torch.load(latest_pth, map_location=device)
                model.load_state_dict(state_dict, strict=True)
                
                resume_global_step = max_step_found
                print(f"[Auto-Resume] Resuming seamlessly from Global Step: {resume_global_step}\n")
        else:
            print(f"[Auto-Resume] Checkpoint directory is empty. Starting fresh from Step 0.")
    else:
        print(f"[Auto-Resume] No checkpoint folder detected. Starting from scratch.")

    # ─── 學習率排程動態對齊 ───
    total_steps = NUM_EPOCHS * len(dataloader)
    warmup_steps = int(total_steps * 0.1)
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 如果是續訓，需要把學習率指針同步推到之前的進度
    if resume_global_step > 0:
        for _ in range(resume_global_step):
            scheduler.step()
        print(f"[Auto-Resume] LR Scheduler synchronized to step {resume_global_step}.")

    # 計算當前起步的 Epoch 以及 Batch 起點
    start_epoch = (resume_global_step // len(dataloader)) + 1
    start_batch_idx = resume_global_step % len(dataloader)
    
    global_step = resume_global_step

    # -------------------------------------------------------------------------
    # 核心 14 天防斷線高頻訓練迴圈
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"🎬 Starting Epoch {epoch}/{NUM_EPOCHS}")
        print("-" * 50)
        
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # 💡 續訓安全機制：如果該 Epoch 內有些 batches 之前已經跑過並存檔了，直接快速跳過
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue
                
            optimizer.zero_grad()
            
            # 將資料送上 GPU
            area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = [t.to(device) for t in batch]
            
            # 擠壓掉 Batch 維度 (batch_size=1)
            area_target = area_target.squeeze(0)
            b2b_conn = b2b_conn.squeeze(0)
            p2b_conn = p2b_conn.squeeze(0)
            pins_pos = pins_pos.squeeze(0)
            metrics = metrics.squeeze(0)
            fp_sol = fp_sol.squeeze(0)
            constraints = constraints.squeeze(0)
            
            block_count = int((area_target != -1).sum().item())
            if block_count > MAX_BLOCKS_LIMIT:
                block_count = MAX_BLOCKS_LIMIT
                
            valid_area = area_target[:block_count].unsqueeze(-1)  
            valid_constraints = constraints[:block_count].float() 
            ground_truth = fp_sol[:block_count]                   
            
            # 對齊座標 (x, y, w, h)
            gt_positions = torch.stack([
                ground_truth[:, 2], ground_truth[:, 3], ground_truth[:, 0], ground_truth[:, 1]
            ], dim=1)
            
            # 擴散加噪
            t_step = torch.randint(0, 1000, (1,)).to(device)
            noise = torch.randn_like(gt_positions) * 5.0
            noised_input = gt_positions + noise
            noised_input_b = noised_input.unsqueeze(0) 
            
            # Forward 推理
            predicted_denoise = model(
                noised_input_b, 
                valid_area, 
                valid_constraints, 
                b2b_conn, 
                t_step, 
                block_count
            )
            
            positions = noised_input + predicted_denoise.squeeze(0)
            
            # 3. 可微分損失函數計算
            loss = compute_training_loss_differentiable(
                positions, b2b_conn, p2b_conn, pins_pos, area_target[:block_count], metrics
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step() # 🚀 保持原 LR 的同時，隨 step 更新排程
            
            global_step += 1
            epoch_loss += loss.item()
            
            # 每一秒/每步即時印出日誌
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] (Global Step {global_step}) -> Loss: {loss.item():.4f}")
            
            # =========================================================================
            # ⚡ 核心修改：每 10 個 batches 就執行一次即時安全存檔
            # =========================================================================
            if global_step % SAVE_FREQ_BATCHES == 0:
                checkpoint_folder.mkdir(exist_ok=True)
                # 檔名加上全域總步數 global_step，以便 Auto-Resume 精準恢復
                checkpoint_name = f"dit_gnn_step_{global_step}_loss_{loss.item():.2f}.pth"
                save_path = checkpoint_folder / checkpoint_name
                
                torch.save(model.state_dict(), save_path)
                print(f"    ⚡ [High-Freq Save] Successfully saved checkpoint to: {save_path}")
                
        # 完滿跑完一輪的日誌
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"📢 Epoch {epoch} Completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # 每輪結束時存一個常規備份（非強制，但可作為大進度標記）
        epoch_save_name = f"dit_gnn_step_{global_step}_epoch_{epoch}_end.pth"
        torch.save(model.state_dict(), checkpoint_folder / epoch_save_name)
        
    print("="*70)
    print("🎉 Edge-GNN + DiT-Small high-frequency training successfully completed!")
    print("="*70)


if __name__ == '__main__':
    main()