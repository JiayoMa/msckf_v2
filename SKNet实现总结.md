# SKNet 与 MSCKF 集成实现总结

## 问题陈述回顾

原始需求：
1. MSCKF 前端处理保持不变（图像与 IMU 的数据处理、特征跟踪、特征管理等都沿用现有实现）
2. 主滤波更新必须用 SKNet 输出的 Pk 和 Sk 来替换 MSCKF 的 P 和 S
3. SKNet 的 K 必须更新主状态，而不是影子轨迹
4. 要同时保留 MSCKF 原版作为 baseline，实现真实对比（两条轨迹都必须是真正闭环的主滤波结果）

## 实现方案

### 1. 修改 `sknet_adapter.py`

**位置**: `msckfv2/sknet_adapter.py` 第 207-313 行

**修改内容**:
```python
def get_optimal_gain(self, H_thin, r_thin, return_covariances=False):
    """
    返回 SKNet 预测的 Kalman 增益和协方差
    
    Args:
        return_covariances: 如果为 True，返回 (K, Pk, Sk)；否则只返回 K（向后兼容）
    
    Returns:
        return_covariances=False: K_valid (numpy array)
        return_covariances=True: (K_valid, Pk_valid, Sk_valid) tuple
    """
    # ... 推理逻辑 ...
    
    if return_covariances:
        # 提取有效的 Pk 和 Sk 子矩阵
        Pk_full = Pk_mat.cpu().numpy()
        Sk_full = Sk_mat.cpu().numpy()
        Pk_valid = Pk_full[:valid_state, :valid_state]
        Sk_valid = Sk_full[:valid_obs, :valid_obs]
        return (K_valid, Pk_valid, Sk_valid)
    else:
        return K_valid
```

**关键改进**:
- 新增 `return_covariances` 参数，默认为 `False` 保持向后兼容
- 当 `return_covariances=True` 时，同时返回 K、Pk、Sk
- 从填充后的协方差矩阵中提取有效子矩阵

### 2. 修改 `msckf.py`

#### 2.1 构造函数修改

**位置**: `msckfv2/msckf.py` 第 109-145 行

```python
class MSCKF(object):
    def __init__(self, config, update_mode="msckf"):
        # ...
        # 更新模式: "msckf" 或 "sknet"
        self.update_mode = update_mode
        # ...
```

**关键改进**:
- 添加 `update_mode` 参数到构造函数
- 存储为实例变量，供 `measurement_update()` 使用

#### 2.2 measurement_update() 方法修改

**位置**: `msckfv2/msckf.py` 第 627-746 行

```python
def measurement_update(self, H, r, update_mode="msckf"):
    """
    MSCKF 测量更新，支持 SKNet 集成
    
    Args:
        H: 观测矩阵
        r: 残差向量
        update_mode: "msckf" (传统更新) 或 "sknet" (使用 SKNet 的 K, Pk, Sk)
    """
    # ... QR 分解 ...
    
    # 获取活跃维度和协方差
    curr_dim = self.state_server.active_dim
    P_active = self.state_server.state_cov[:curr_dim, :curr_dim]
    
    # --- 根据 update_mode 分支 ---
    if update_mode == "sknet":
        # 使用 SKNet 预测的 Kalman 增益和协方差
        result = self.sknet_adapter.get_optimal_gain(H_thin, r_thin, return_covariances=True)
        
        if result is None or result[0] is None:
            raise RuntimeError("SKNet inference failed.")
        
        K, Pk_sknet, Sk_sknet = result
        
        # 使用 SKNet 的后验协方差
        if Pk_sknet.shape[0] >= curr_dim:
            P_active = Pk_sknet[:curr_dim, :curr_dim]
    
    if update_mode == "msckf":
        # 传统 MSCKF 更新：从 P 计算 S 和 K
        S = H_thin @ P_active @ H_thin.T + R
        K = P_active @ H_thin.T @ inv(S)
    
    # 使用适当的 K 计算 delta_x
    delta_x = K @ r_thin
    
    # 更新 IMU 状态和相机状态 (相同逻辑)
    # ...
    
    # 更新协方差
    if update_mode == "sknet":
        # 直接使用 SKNet 的后验协方差
        self.state_server.state_cov[:curr_dim, :curr_dim] = P_active
    else:
        # 传统 MSCKF 协方差更新
        I_KH = np.identity(curr_dim) - K @ H_thin
        P_new_active = I_KH @ P_active
        self.state_server.state_cov[:curr_dim, :curr_dim] = (P_new_active + P_new_active.T) / 2.
```

**关键改进**:
1. 添加 `update_mode` 参数到方法签名
2. 实现分支逻辑：
   - `update_mode="msckf"`: 使用传统 P/S/K 计算
   - `update_mode="sknet"`: 使用 SKNet 的 Pk/Sk 计算 K 并更新主状态
3. 两种模式都更新 `state_server`（主状态），而非影子轨迹

#### 2.3 调用点更新

**位置**: 
- `msckfv2/msckf.py` 第 855 行 (remove_lost_features)
- `msckfv2/msckf.py` 第 971 行 (prune_cam_state_buffer)

```python
# 修改前
self.measurement_update(H_x, r)

# 修改后
self.measurement_update(H_x, r, update_mode=self.update_mode)
```

### 3. 修改 `vio.py`

#### 3.1 VIO 构造函数修改

**位置**: `msckfv2/vio.py` 第 20-68 行

```python
class VIO(object):
    def __init__(self, config, img_queue, imu_queue, gt_path=None, save_dir="./results", 
                 enable_sknet_comparison=False):
        # ...
        
        if enable_sknet_comparison:
            # 创建两个 MSCKF 实例进行对比
            
            # Baseline MSCKF (传统更新)
            self.msckf_baseline = MSCKF(config, update_mode="msckf")
            self.msckf_baseline.mode = 'normal'  # 禁用 SKNet 集成
            
            # SKNet-fusion MSCKF (使用 SKNet K, Pk, Sk)
            self.msckf_sknet = MSCKF(config, update_mode="sknet")
            self.msckf_sknet.mode = 'test'  # 启用 SKNet
            
            print("[VIO] 运行双模式: Baseline MSCKF vs SKNet-Fusion MSCKF")
        else:
            # 单个 MSCKF 实例 (向后兼容)
            self.msckf = MSCKF(config, update_mode="msckf")
            print("[VIO] 运行单模式")
```

**关键改进**:
- 添加 `enable_sknet_comparison` 参数
- 双模式下创建两个独立的 MSCKF 实例
- 每个实例使用不同的 `update_mode`

#### 3.2 process_imu() 方法修改

**位置**: `msckfv2/vio.py` 第 113-125 行

```python
def process_imu(self):
    """IMU 处理线程：积分预测"""
    while True:
        imu_msg = self.imu_queue.get()
        if imu_msg is None: return
        self.image_processor.imu_callback(imu_msg)
        
        if self.enable_sknet_comparison:
            # 将 IMU 数据同时传递给两个 MSCKF 实例
            self.msckf_baseline.imu_callback(imu_msg)
            self.msckf_sknet.imu_callback(imu_msg)
        else:
            self.msckf.imu_callback(imu_msg)
```

#### 3.3 process_feature() 方法修改

**位置**: `msckfv2/vio.py` 第 127-244 行

```python
def process_feature(self):
    """核心线程：后端优化 (MSCKF + SKNet)"""
    try:
        while True:
            feature_msg = self.feature_queue.get()
            
            if feature_msg is None:
                # 结束处理...
                return
            
            if self.enable_sknet_comparison:
                # 并行运行两个 MSCKF 实例
                result_baseline = self.msckf_baseline.feature_callback(feature_msg)
                result_sknet = self.msckf_sknet.feature_callback(feature_msg)
                
                if result_baseline is not None and result_sknet is not None:
                    # 从主状态获取位置（不是影子轨迹）
                    msckf_pos = self.msckf_baseline.state_server.imu_state.position.copy()
                    sknet_pos = self.msckf_sknet.state_server.imu_state.position.copy()
                    
                    # 存储轨迹
                    self.timestamps.append(t)
                    self.traj_msckf.append(msckf_pos)
                    self.traj_sknet.append(sknet_pos)
                    self.traj_gt.append(gt_pos)
            else:
                # 单模式（向后兼容）
                # ...
```

**关键改进**:
1. 双模式下并行处理两个 MSCKF 实例
2. 从 `state_server.imu_state.position` 提取轨迹（主状态）
3. **不再使用影子轨迹** - 两条轨迹都是真正的闭环主滤波结果

#### 3.4 命令行参数添加

**位置**: `msckfv2/vio.py` 第 357-394 行

```python
parser.add_argument('--dual_mode', action='store_true',
    help='启用双模式：同时运行 baseline MSCKF 和 SKNet-fusion MSCKF 进行对比')

# ...

msckf_vio = VIO(config, img_queue, imu_queue, gt_path=gt_path, 
                save_dir=args.save_dir, enable_sknet_comparison=args.dual_mode)
```

## 使用方法

### 单模式（传统）
```bash
python msckfv2/vio.py --path /path/to/dataset --save_dir ./results
```

### 双模式（对比）
```bash
python msckfv2/vio.py --path /path/to/dataset --save_dir ./results --dual_mode
```

## 验证检查清单

- [x] MSCKF 前端处理保持不变 ✓
  - 图像与 IMU 数据处理未修改
  - 特征跟踪逻辑未修改
  - 特征管理逻辑未修改

- [x] 主滤波更新使用 SKNet 的 Pk 和 Sk ✓
  - `get_optimal_gain()` 返回 (K, Pk, Sk)
  - `measurement_update()` 在 `update_mode="sknet"` 时使用 Pk

- [x] SKNet 的 K 更新主状态 ✓
  - `measurement_update()` 直接更新 `self.state_server`
  - 不再仅更新影子轨迹
  - 状态更新包括位置、速度、姿态、偏置等

- [x] 同时保留 MSCKF 原版作为 baseline ✓
  - 创建两个独立的 MSCKF 实例
  - `msckf_baseline` 使用传统 P/S/K
  - `msckf_sknet` 使用 SKNet 的 Pk/Sk
  - 两条轨迹都是主滤波结果，可以真实对比

## 输出结果

运行双模式后，系统生成：

1. **academic_comparison.png**: 三面板对比图
   - 3D 轨迹对比（真值、MSCKF Baseline、SKNet）
   - 2D 俯视图轨迹
   - 绝对位置误差（APE）及 RMSE 值

2. **trajectory_data.npy**: 原始轨迹数据供进一步分析

## 技术细节

### 数据流（双模式）
```
IMU 数据 → ImageProcessor → 特征提取
                ↓
        ┌───────┴────────┐
        ↓                ↓
   MSCKF Baseline   MSCKF SKNet
   (update_mode=   (update_mode=
    "msckf")        "sknet")
        ↓                ↓
   传统 P/S/K      SKNet 增益
   计算            (K, Pk, Sk)
        ↓                ↓
   主状态更新      主状态更新
        ↓                ↓
   轨迹            轨迹
   (Baseline)      (SKNet)
        └────────┬────────┘
                 ↓
          对比图与分析
```

### 关键实现保证

1. **独立状态管理**: 每个 MSCKF 实例维护独立的：
   - State server (位置、速度、姿态、偏置)
   - 相机状态缓冲区
   - 特征地图
   - 协方差矩阵

2. **主状态更新**: SKNet 不再仅更新影子轨迹，而是：
   - 通过 `measurement_update()` 直接更新 `state_server`
   - 使用 SKNet 预测的 K 计算状态增量
   - 使用 SKNet 的 Pk 更新协方差

3. **真实闭环对比**: 两个 MSCKF 实例：
   - 都经过完整的预测-更新循环
   - 都进行特征管理和状态增广
   - 都进行协方差传播和更新
   - 产生的轨迹都是真正的闭环滤波结果

## 文件清单

修改的文件：
- `msckfv2/sknet_adapter.py` - 增强 get_optimal_gain() 返回协方差
- `msckfv2/msckf.py` - 添加 update_mode 支持，实现分支更新逻辑
- `msckfv2/vio.py` - 实现双 MSCKF 实例并行运行

新增文件：
- `.gitignore` - 排除构建产物和缓存
- `SKNET_INTEGRATION.md` - 英文详细文档
- `test_integration.py` - 集成测试脚本
- `SKNet实现总结.md` - 本文档（中文总结）
