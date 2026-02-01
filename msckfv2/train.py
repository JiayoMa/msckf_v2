import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

from dnn import DNN_SKalmanNet_GSS 

# ==========================================
# 1. 数据集定义 (处理数据加载、对齐和特征构造)
# ==========================================
class VioSknetDataset(Dataset):
    def __init__(self, data_dir, gt_csv_path, x_dim=147):
        self.x_dim = x_dim
        self.samples = []
        
        print(f"[Dataset] Loading VIO data from {data_dir}...")
        vio_data = self._load_vio_data(data_dir)
        
        print(f"[Dataset] Loading Ground Truth from {gt_csv_path}...")
        gt_df = self._load_gt(gt_csv_path)
        
        print("[Dataset] Aligning and preprocessing features...")
        self._preprocess_data(vio_data, gt_df)
        
        print(f"[Dataset] Ready. Total samples: {len(self.samples)}")

    def _load_vio_data(self, data_dir):
        """加载所有 batch_*.npy 文件"""
        file_list = glob.glob(os.path.join(data_dir, "batch_*.npy"))
        # 按文件名数字排序，保证时序正确
        file_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
        
        all_data = []
        for f in file_list:
            data = np.load(f, allow_pickle=True)
            all_data.extend(data)
        return all_data

    def _load_gt(self, gt_path):
        """加载 EuRoC 真值 CSV"""
        df = pd.read_csv(gt_path)
        # 清理列名空格
        df.columns = [c.strip() for c in df.columns]
        # 纳秒 -> 秒
        df['#timestamp'] = df['#timestamp'] / 1e9
        return df

    def _get_interpolated_gt(self, timestamp, gt_df):
        """根据时间戳插值获取真值位置"""
        # 找到相邻的两个时间点
        # 这里为了速度，假设 GT 是高频且覆盖 VIO 时间的
        # 使用 pandas 的 asof 或者 searchsorted
        idx = np.searchsorted(gt_df['#timestamp'], timestamp)
        
        if idx == 0 or idx >= len(gt_df):
            return None # 时间戳越界

        t0 = gt_df.iloc[idx-1]['#timestamp']
        t1 = gt_df.iloc[idx]['#timestamp']
        
        # 线性插值因子
        alpha = (timestamp - t0) / (t1 - t0)
        
        p0 = gt_df.iloc[idx-1][['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].to_numpy(dtype=float)
        p1 = gt_df.iloc[idx][['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].to_numpy(dtype=float)
        
        pos_interp = p0 + alpha * (p1 - p0)
        return pos_interp

    def _pad_matrix(self, mat, target_dim):
        """Padding 矩阵到 147x147"""
        rows, cols = mat.shape
        pad = np.zeros((target_dim, target_dim))
        v_r = min(rows, target_dim)
        v_c = min(cols, target_dim)
        pad[:v_r, :v_c] = mat[:v_r, :v_c]
        return pad

    def _pad_vector(self, vec, target_dim):
        """Padding 向量到 147x1"""
        if vec.ndim == 1: vec = vec.reshape(-1, 1)
        rows = vec.shape[0]
        pad = np.zeros((target_dim, 1))
        v_r = min(rows, target_dim)
        pad[:v_r, :] = vec[:v_r, :]
        return pad

    def _preprocess_data(self, vio_data, gt_df):
        """
        核心逻辑：
        1. 遍历 VIO 帧
        2. 找到对应 GT
        3. 对齐坐标原点 (Zero-Alignment)
        4. 构造 RNN 所需的时序差分特征
        """
        # 历史变量初始化
        state_post_past = np.zeros((self.x_dim, 1))
        obs_past = np.zeros((self.x_dim, 1))
        
        # 坐标对齐变量
        vio_start_pos = None
        gt_start_pos = None

        for i, frame in enumerate(vio_data):
            ts = frame['timestamp']
            
            # 1. 获取真值
            gt_pos = self._get_interpolated_gt(ts, gt_df)
            if gt_pos is None: continue # 跳过没有真值的帧
            
            # 2. 获取 VIO 先验状态 (Prediction)
            state_pred = frame['state_pred'] # (147, 1)
            # 提取 VIO 位置 (索引 12,13,14)
            vio_pos_pred = state_pred[12:15].flatten()

            # 3. 坐标系对齐 (第一帧对齐)
            if vio_start_pos is None:
                vio_start_pos = vio_pos_pred.copy()
                gt_start_pos = gt_pos.copy()
            
            # 相对位移 (VIO系 vs GT系)
            # 我们希望: (VIO_Pred - VIO_Start) + Correction ≈ (GT - GT_Start)
            # 所以 Target_Pos = (GT - GT_Start) + VIO_Start
            # 这样就把 GT 搬运到了 VIO 的坐标系下
            target_pos_aligned = (gt_pos - gt_start_pos) + vio_start_pos

            # 4. 准备网络输入 (Padding)
            H = frame['H']
            r = frame['r']
            H_pad = self._pad_matrix(H, self.x_dim)
            r_pad = self._pad_vector(r, self.x_dim)
            
            # 5. 构造差分特征
            # 近似: 假设上一帧 Update 后 ≈ 当前帧 Predict (Teacher Forcing 近似)
            # 因为我们没有记录真实的 Post，且离线训练没有闭环
            if i == 0:
                state_post_past = state_pred.copy()
            
            # Features
            state_inno = state_post_past - state_pred # x_{k-1|k-1} - x_{k-1|k-2} (这里会有时序错位，但在Open-Loop训练中常用)
            # 更合理的 Open-loop 近似: state_inno = 0 (假设上一帧完美) 或者使用 state_pred 的一阶差分
            # 这里我们使用 state_pred 的差分作为 diff_state
            diff_state = state_pred - state_post_past
            diff_obs = r_pad - obs_past
            lin_error = r_pad
            
            # 6. 保存样本
            self.samples.append({
                'features': {
                    'state_inno': np.zeros_like(diff_state), # 简化处理，避免噪声
                    'diff_state': diff_state.astype(np.float32),
                    'diff_obs': diff_obs.astype(np.float32),
                    'lin_error': lin_error.astype(np.float32),
                    'H': H_pad.astype(np.float32),
                    'r': r_pad.astype(np.float32)
                },
                'meta': {
                    'vio_pred_pos': vio_pos_pred.astype(np.float32),
                    'gt_pos': target_pos_aligned.astype(np.float32) # 已对齐的真值
                }
            })
            
            # 更新历史
            state_post_past = state_pred.copy()
            obs_past = r_pad.copy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==========================================
# 2. 训练主程序
# ==========================================
def train():
    # --- 配置参数 ---
    # 采集数据的目录
    DATA_DIR = "./train_data_qr" 
    # 真值文件路径 (请修改为你自己的路径)
    GT_PATH = "/home/qiuying/Downloads/kalmannet/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    
    # 训练超参
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_PATH = "model_best.pt"
    
    # 1. 检查路径
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} is empty or does not exist.")
        print("Please run 'python vio.py' in 'train' mode first.")
        return

    # 2. 加载数据集
    dataset = VioSknetDataset(DATA_DIR, GT_PATH, x_dim=147)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 3. 初始化模型
    # 注意: dnn.py 必须已修改为固定 hidden_dim，否则这里会爆显存
    model = DNN_SKalmanNet_GSS(x_dim=147, y_dim=147)
    model.to(DEVICE)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    print(f"Start training on {DEVICE}...")
    loss_history = []

    for epoch in range(EPOCHS):
        total_loss = 0
        
        for i, batch in enumerate(dataloader):
            # 提取数据
            feats = batch['features']
            meta = batch['meta']
            
            # --- [修改 1] 动态获取当前 Batch 大小 ---
            # 最后一个 batch 可能不足 64，所以不能用全局 BATCH_SIZE
            curr_b = feats['state_inno'].shape[0] 
            
            # 转移到 GPU 并转为 float32
            # --- [修改 2] 将所有的 BATCH_SIZE 替换为 curr_b ---
            state_inno = feats['state_inno'].to(DEVICE).float().view(curr_b, -1)
            diff_state = feats['diff_state'].to(DEVICE).float().view(curr_b, -1)
            diff_obs   = feats['diff_obs'].to(DEVICE).float().view(curr_b, -1)
            lin_error  = feats['lin_error'].to(DEVICE).float().view(curr_b, -1)
            H          = feats['H'].to(DEVICE).float().view(curr_b, -1) 
            r          = feats['r'].to(DEVICE).float().view(curr_b, -1)
            
            # 前向传播
            Pk, Sk = model(state_inno, r, diff_state, diff_obs, lin_error, H)
            
            # --- 计算 Loss ---
            # 1. 恢复矩阵形状
            # --- [修改 3] 这里也要用 curr_b ---
            Pk_mat = Pk.view(curr_b, 147, 147)
            Sk_mat = Sk.view(curr_b, 147, 147)
            H_mat  = feats['H'].to(DEVICE).float() # (curr_b, 147, 147)
            r_vec  = feats['r'].to(DEVICE).float().view(curr_b, 147, 1)
            
            # 2. 计算 Kalman Gain
            K = torch.bmm(torch.bmm(Pk_mat, H_mat.transpose(1, 2)), Sk_mat)
            
            # 3. 计算修正量
            delta_x = torch.bmm(K, r_vec).squeeze(2) # (curr_b, 147)
            
            # 4. 提取位置修正量 (Indices 12, 13, 14)
            pos_correction = delta_x[:, 12:15]
            
            # 5. 计算最终预测位置: Pos_Final = Pos_Prior + Correction
            pos_prior = meta['vio_pred_pos'].to(DEVICE)
            pos_final = pos_prior + pos_correction
            
            # 6. 计算 Loss: MSE(Pos_Final, Pos_GT)
            gt_pos = meta['gt_pos'].to(DEVICE)
            loss = loss_fn(pos_final, gt_pos)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 (防止 RNN 梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 保存最佳模型
        if epoch == 0 or avg_loss < min(loss_history[:-1]):
            torch.save({'state_dict': model.state_dict()}, SAVE_PATH)
            # print("  -> Model saved.")

    print("Training Complete.")
    
    # 绘制 Loss 曲线
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss (Position MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (m^2)")
    plt.savefig("train_loss.png")
    print("Loss curve saved as train_loss.png")

if __name__ == "__main__":
    train()