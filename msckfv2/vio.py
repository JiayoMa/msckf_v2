import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from queue import Queue
from threading import Thread

# 引入必要的绘图库
from mpl_toolkits.mplot3d import Axes3D

# 自定义模块引入
from config import ConfigEuRoC
from image import ImageProcessor
from msckf import MSCKF

class VIO(object):
    def __init__(self, config, img_queue, imu_queue, gt_path=None, save_dir="./results"):
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.trajectory = []
        
        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config)

        # --- 数据容器 ---
        self.traj_msckf = []
        self.traj_sknet = []
        self.traj_gt = [] 
        self.timestamps = []
        
        # --- 加载真值 (Ground Truth) ---
        self.gt_df = None
        if gt_path and os.path.exists(gt_path):
            print(f"[VIO] Loading ground truth from {gt_path}")
            self.gt_df = pd.read_csv(gt_path)
            # 清理列名空格
            self.gt_df.columns = [c.strip() for c in self.gt_df.columns]
            # 转换时间戳 (ns -> s)
            self.gt_df['#timestamp'] = self.gt_df['#timestamp'] / 1e9
        else:
            print(f"[VIO] Warning: Ground truth not found at {gt_path}")

        # --- 启动线程 ---
        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        
        # 设置为守护线程 (可选，但在 os._exit 下不是必须的)
        self.img_thread.daemon = True
        self.imu_thread.daemon = True
        self.vio_thread.daemon = True

        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()
        
        print("[VIO] System Initialized. Threads started.")

    def get_gt_pose_at_time(self, timestamp):
        """查找最近邻的真值位置"""
        if self.gt_df is None: return np.zeros(3)
        # 找到时间戳最近的一行
        idx = (self.gt_df['#timestamp'] - timestamp).abs().idxmin()
        row = self.gt_df.iloc[idx]
        try:
            return np.array([row['p_RS_R_x [m]'], row['p_RS_R_y [m]'], row['p_RS_R_z [m]']])
        except KeyError:
            return np.zeros(3)

    def process_img(self):
        """图像处理线程：特征提取"""
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return
            
            # [监控] 如果特征队列堆积，说明后端处理过慢或卡死
            if self.feature_queue.qsize() > 50:
                print(f"[VIO Warning] Feature Queue clogged ({self.feature_queue.qsize()}). Backend might be stuck.")

            feature_msg = self.image_processor.stareo_callback(img_msg)
            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        """IMU 处理线程：积分预测"""
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None: return
            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)

    def process_feature(self):
        """核心线程：后端优化 (MSCKF + SKNet)"""
        try:
            while True:
                feature_msg = self.feature_queue.get()
                
                # --- 1. 正常结束检查 ---
                if feature_msg is None:
                    print("[VIO] Dataset finished. Finalizing...")
                    if hasattr(self.msckf, 'finalize'):
                        self.msckf.finalize()
                    
                    self.save_comparison_plot() 
                    self.save_trajectory() 
                    print("[VIO] Process finished successfully.")
                    os._exit(0) # 正常退出
                    return
                
                # --- 2. 执行更新 ---
                # 这里会调用 Adapter，若 SKNet 失败会抛出 RuntimeError
                result = self.msckf.feature_callback(feature_msg)

                if result is not None:
                    t = feature_msg.timestamp
                    
                    # --- [A] 获取 Baseline (MSCKF) ---
                    msckf_pos = self.msckf.state_server.imu_state.position.copy()
                    
                    # --- [B] 获取 SKNet (Strict Mode) ---
                    sknet_pos = None
                    
                    # 检查 Adapter 状态
                    if hasattr(self.msckf, 'sknet_adapter') and self.msckf.sknet_adapter is not None:
                        # 获取当前的 shadow state
                        if self.msckf.sknet_adapter.shadow_pos is not None:
                            sknet_pos = self.msckf.sknet_adapter.shadow_pos.copy()
                    
                    # --- [C] 严格终止检查 ---
                    # 如果 MSCKF 成功了，但 SKNet 没有值，说明 SKNet 刚刚挂了
                    if sknet_pos is None:
                        err_msg = f"[VIO Critical] SKNet Lost Tracking at t={t:.3f} (shadow_pos is None)."
                        print(f"\033[91m{err_msg}\033[0m")
                        raise RuntimeError(err_msg) # 抛出异常进入 except 块

                    # --- [D] 获取真值 ---
                    gt_pos = self.get_gt_pose_at_time(t)
                    
                    # 存入列表
                    self.timestamps.append(t)
                    self.traj_msckf.append(msckf_pos)
                    self.traj_sknet.append(sknet_pos)
                    self.traj_gt.append(gt_pos)
                
                    self.trajectory.append(result.cam0_pose)
                    
                    # 可选：每 100 帧打印一次进度
                    if len(self.timestamps) % 100 == 0:
                        print(f"[VIO] Processed {len(self.timestamps)} frames. t={t:.2f}")

        except Exception as e:
            # --- 异常捕获区 ---
            print("\n" + "!"*50)
            print("[CRITICAL ERROR] VIO Thread crashed!")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            print("-" * 20 + " Traceback " + "-" * 20)
            traceback.print_exc()
            print("!"*50 + "\n")
            
            print("[VIO] Attempting emergency save of trajectory data...")
            try:
                self.save_comparison_plot()
                self.save_trajectory()
            except Exception as save_err:
                print(f"[VIO] Emergency save failed: {save_err}")
            
            print("[VIO] Exiting with error code 1.")
            os._exit(1) # 强制杀死所有线程

    def save_comparison_plot(self):
        """绘制学术对比图"""
        if len(self.timestamps) < 2:
            print("[VIO] Not enough data to plot.")
            return

        try:
            msckf = np.array(self.traj_msckf)
            sknet = np.array(self.traj_sknet)
            gt = np.array(self.traj_gt)
            
            # --- 原点对齐 (Zero-Start Alignment) ---
            if len(msckf) > 0: msckf = msckf - msckf[0]
            if len(sknet) > 0: sknet = sknet - sknet[0]
            if len(gt) > 0: gt = gt - gt[0] 

            # 创建画布
            fig = plt.figure(figsize=(20, 6))
            
            # 1. 3D Trajectory
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            ax1.plot(gt[:,0], gt[:,1], gt[:,2], 'k--', label='Ground Truth')
            ax1.plot(msckf[:,0], msckf[:,1], msckf[:,2], 'b-', label='MSCKF (Baseline)', alpha=0.6)
            ax1.plot(sknet[:,0], sknet[:,1], sknet[:,2], 'r-', label='SKNet (Ours)', linewidth=2)
            ax1.set_title("3D Trajectory")
            ax1.set_xlabel("X [m]")
            ax1.set_ylabel("Y [m]")
            ax1.set_zlabel("Z [m]")
            ax1.legend()

            # 2. 2D Plane (X-Y)
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.plot(gt[:,0], gt[:,1], 'k--', label='Ground Truth')
            ax2.plot(msckf[:,0], msckf[:,1], 'b-', label='MSCKF')
            ax2.plot(sknet[:,0], sknet[:,1], 'r-', label='SKNet')
            ax2.set_title("2D Trajectory (Top View)")
            ax2.set_xlabel("X [m]")
            ax2.set_ylabel("Y [m]")
            ax2.axis('equal')
            ax2.grid(True)
            ax2.legend()
            
            # 3. APE Error (Position Error)
            ax3 = fig.add_subplot(1, 3, 3)
            err_msckf = np.linalg.norm(msckf - gt, axis=1)
            err_sknet = np.linalg.norm(sknet - gt, axis=1)
            
            # 计算 RMSE
            rmse_msckf = np.sqrt(np.mean(err_msckf**2))
            rmse_sknet = np.sqrt(np.mean(err_sknet**2))

            t_axis = np.array(self.timestamps) - self.timestamps[0]
            
            ax3.plot(t_axis, err_msckf, 'b-', label=f'MSCKF (RMSE={rmse_msckf:.2f}m)')
            ax3.plot(t_axis, err_sknet, 'r-', label=f'SKNet (RMSE={rmse_sknet:.2f}m)')
            ax3.set_title("Absolute Position Error (APE)")
            ax3.set_xlabel("Time [s]")
            ax3.set_ylabel("Error [m]")
            ax3.grid(True)
            ax3.legend()
            
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, "academic_comparison.png")
            plt.savefig(save_path, dpi=150)
            print(f"[VIO] Comparison plot saved to {save_path}")
        except Exception as e:
            print(f"[VIO] Error plotting: {e}")
            traceback.print_exc()

    def save_trajectory(self):
        """保存原始数据到 npy 文件"""
        if not self.timestamps: return
        data_path = os.path.join(self.save_dir, 'trajectory_data.npy')
        
        # 确保所有数组长度一致
        min_len = min(len(self.timestamps), len(self.traj_msckf), len(self.traj_sknet), len(self.traj_gt))
        
        np.save(data_path, {
            'timestamps': np.array(self.timestamps[:min_len]),
            'msckf_traj': np.array(self.traj_msckf[:min_len]),
            'sknet_traj': np.array(self.traj_sknet[:min_len]),
            'gt_traj': np.array(self.traj_gt[:min_len])
        })
        print(f"[VIO] Trajectory data saved to {data_path}")

if __name__ == '__main__':
    from dataset import EuRoCDataset, DataPublisher
    
    # --- 参数设置 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/qiuying/Downloads/kalmannet/machine_hall/MH_01_easy', 
        help='Path of EuRoC MAV dataset.')
    parser.add_argument('--save_dir', type=str, default='./results', 
        help='Directory to save results.')
    args = parser.parse_args()

    # 路径检查
    if not os.path.exists(args.path):
        print(f"Error: Dataset path {args.path} does not exist.")
        exit(1)

    # --- 1. 数据集加载 (关键修正: offset=0) ---
    print(f"[Main] Loading dataset from {args.path}...")
    dataset = EuRoCDataset(args.path)
    # [CRITICAL FIX] 从 0 开始，让 Filter 有机会在静止状态下完成初始化 (Align gravity)
    # 如果从 40 开始，Filter 会在运动中启动，导致重力对齐失败，轨迹直接飞掉
    dataset.set_starttime(offset=40.) 

    # --- 2. 队列与 Config 初始化 ---
    img_queue = Queue()
    imu_queue = Queue()
    config = ConfigEuRoC()
    
    # 构建 GT 路径 (兼容两种目录结构)
    gt_path = os.path.join(args.path, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
    if not os.path.exists(gt_path):
        gt_path_alt = os.path.join(args.path, 'state_groundtruth_estimate0', 'data.csv')
        if os.path.exists(gt_path_alt):
            gt_path = gt_path_alt

    # --- 3. 初始化 VIO 系统 ---
    msckf_vio = VIO(config, img_queue, imu_queue, gt_path=gt_path, save_dir=args.save_dir)

    # --- 4. 启动数据发布器 ---
    duration = float('inf')
    ratio = 0.4 # 控制播放速度
    
    print("[Main] Starting Data Publishers...")
    imu_publisher = DataPublisher(dataset.imu, imu_queue, duration, ratio)
    img_publisher = DataPublisher(dataset.stereo, img_queue, duration, ratio)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
    
    # 主线程可以做点别的，或者就空转等待子线程结束
    # 由于 VIO 线程里有 os._exit，这里不需要复杂的 join 逻辑
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Main] Keyboard Interrupt. Exiting...")
        os._exit(0)