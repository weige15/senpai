import torch
import os
import glob

def check_all_checkpoints(folder_path):
    # 抓取資料夾下所有的 .pth 檔案
    pth_files = glob.glob(os.path.join(folder_path, '*.pth'))
    
    if not pth_files:
        print(f"在 {folder_path} 找不到任何 .pth 檔案。")
        return

    print(f"找到 {len(pth_files)} 個 checkpoint，開始讀取...\n")
    
    results = []
    no_loss_files = []

    for file_path in pth_files:
        filename = os.path.basename(file_path)
        try:
            # 加上 weights_only=False 才能讀取完整的 dict
            # map_location='cpu' 避免因為 GPU 記憶體不足而卡死
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict):
                # 嘗試尋找 'loss' 或 'val_loss' 等關鍵字
                loss_val = checkpoint.get('loss') or checkpoint.get('val_loss')
                
                if loss_val is not None:
                    # 如果 loss 是 tensor，轉成純數值
                    if isinstance(loss_val, torch.Tensor):
                        loss_val = loss_val.item()
                    results.append({'file': filename, 'loss': float(loss_val)})
                else:
                    no_loss_files.append(filename)
            else:
                no_loss_files.append(filename)
                
        except Exception as e:
            print(f"讀取 {filename} 失敗: {e}")

    # 針對有找到 loss 的檔案，依據 loss 數值由小到大排序 (遞增)
    results.sort(key=lambda x: x['loss'])

    # --- 印出結果 ---
    print("=== 🏆 最佳 Loss 排行榜 (由小到大) ===")
    for rank, res in enumerate(results, 1):
        print(f"[{rank:02d}] Loss: {res['loss']:.5f} | File: {res['file']}")

    print("\n------------------------------------------------")
    if no_loss_files:
        print(f"ℹ️ 有 {len(no_loss_files)} 個檔案未記錄 loss (可能只存了 model_state_dict):")
        # 如果太多，只印前 5 個當作代表
        for f in no_loss_files[:5]:
            print(f"  - {f}")
        if len(no_loss_files) > 5:
            print("  - ... (下略)")

if __name__ == '__main__':
    # 將路徑指向你的 checkpoints 資料夾
    check_all_checkpoints('checkpoints_fixed/')
