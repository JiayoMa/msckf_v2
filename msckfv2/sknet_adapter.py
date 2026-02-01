import numpy as np
import torch
import torch.nn as nn
import traceback
import sys
import copy
from utils import to_rotation, to_quaternion, small_angle_quaternion, quaternion_multiplication

# --- Import DNN Module Safely ---
try:
    from dnn import DNN_SKalmanNet_GSS 
except ImportError:
    print("[SKNetAdapter] Warning: Could not import DNN_SKalmanNet_GSS. Check your python path.")
    # Placeholder to prevent crash during import, but will crash at runtime if used
    class DNN_SKalmanNet_GSS(nn.Module):
        def __init__(self, x, y): 
            super().__init__()
            raise ImportError("DNN_SKalmanNet_GSS module is missing.")
        def forward(self, *args, **kwargs):
            raise ImportError("DNN_SKalmanNet_GSS module is missing.")

class SKNetAdapter:
    def __init__(self, config, model_path=None, device='cuda'):
        self.config = config
        self.device = device
        
        # [Config] Dimensions (147 = 21 IMU + 21 Camera states * 6 + Padding)
        # 即使实际相机状态少于21个，我们也在 _pad_qr_data 中处理
        self.x_dim = 147 
        self.y_dim = 147 
        
        # [Model] Load Network
        try:
            self.kf_net = DNN_SKalmanNet_GSS(self.x_dim, self.y_dim)
            self.kf_net.to(self.device)
        except Exception as e:
            print(f"[SKNetAdapter] Critical Error during initialization: {e}")
            self.kf_net = None
        
        if model_path and self.kf_net is not None:
            self.load_model(model_path)
        elif self.kf_net is not None:
            print("[SKNetAdapter] Warning: No model path provided. Running with random weights (Expect divergence).")

        # [Shadow State] Independent variables for the Neural Network Trajectory
        self.shadow_pos = np.zeros(3)
        self.shadow_vel = np.zeros(3)
        self.shadow_q   = np.array([0., 0., 0., 1.]) # quaternion [x,y,z,w]
        self.shadow_bg  = np.zeros(3)
        self.shadow_ba  = np.zeros(3)
        self.is_initialized = False

        # [History] Variables for Temporal Features
        self.state_post_past = None 
        self.state_pred_past = None
        self.obs_past = None 
        self.current_pred_temp = None
        
        # 用于计算传播增量的上一帧 MSCKF 状态
        self.msckf_post_past_for_prop = None 
        self.first_run = True

    def load_model(self, model_path):
        if self.kf_net is None: return

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle both full model save and state_dict save
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            
            if isinstance(state_dict, nn.Module):
                self.kf_net = state_dict
                self.kf_net.to(self.device)
            else:
                self.kf_net.load_state_dict(state_dict)
                
            self.kf_net.eval()
            # Reset hidden states if supported
            if hasattr(self.kf_net, 'initialize_hidden'):
                self.kf_net.initialize_hidden()
            
            print(f"[SKNetAdapter] Loaded model from {model_path}")
        except Exception as e:
            print(f"[SKNetAdapter] Load failed: {e}")
            traceback.print_exc()

    def initialize_shadow_state(self, msckf_imu_state):
        """Called once when VIO initializes to align starting points."""
        self.shadow_pos = msckf_imu_state.position.copy()
        self.shadow_vel = msckf_imu_state.velocity.copy()
        self.shadow_q   = msckf_imu_state.orientation.copy()
        self.shadow_bg  = msckf_imu_state.gyro_bias.copy()
        self.shadow_ba  = msckf_imu_state.acc_bias.copy()
        
        # 记录初始状态，防止第一帧没有参考
        self.current_pred_temp = msckf_imu_state.copy()
        self.msckf_post_past_for_prop = copy.deepcopy(msckf_imu_state)
        
        self.is_initialized = True
        print("[SKNetAdapter] Shadow state initialized.")

    def flatten_msckf_state(self, state_server_or_imu):
        """
        Flattens MSCKF state into (147, 1) vector for NN input.
        兼容输入是 state_server 或者 纯 imu_state
        """
        full_state = np.zeros(self.x_dim)
        
        # 1. Handle Input Type
        imu = None
        cam_states = {}
        
        if hasattr(state_server_or_imu, 'imu_state'):
            imu = state_server_or_imu.imu_state
            cam_states = state_server_or_imu.cam_states
        elif hasattr(state_server_or_imu, 'position'):
            imu = state_server_or_imu
        else:
            # Fallback if raw vector
            if isinstance(state_server_or_imu, np.ndarray):
                return state_server_or_imu.reshape(-1, 1)
            return np.zeros((self.x_dim, 1))

        # 2. Fill IMU State (Indices 0-21)
        # Order: [Rot(3), Bg(3), Vel(3), Ba(3), Pos(3), Rot_Ex(3), Pos_Ex(3)]
        full_state[0:3]   = to_rotation(imu.orientation).flatten()[:3] # Error state approximation
        full_state[3:6]   = imu.gyro_bias
        full_state[6:9]   = imu.velocity
        full_state[9:12]  = imu.acc_bias
        full_state[12:15] = imu.position
        full_state[15:18] = to_rotation(to_quaternion(imu.R_imu_cam0)).flatten()[:3]
        full_state[18:21] = imu.t_cam0_imu
        
        # 3. Fill Camera States (Indices 21-147)
        # Must sort keys to ensure deterministic order
        sorted_cam_ids = sorted(list(cam_states.keys()))
        for i, cam_id in enumerate(sorted_cam_ids):
            start_idx = 21 + i * 6
            if start_idx + 6 > self.x_dim: 
                break
            cam = cam_states[cam_id]
            full_state[start_idx : start_idx+3]   = to_rotation(cam.orientation).flatten()[:3]
            full_state[start_idx+3 : start_idx+6] = cam.position
            
        return full_state.reshape(-1, 1)

    def _pad_qr_data(self, H_thin, r_thin):
        """
        Pads H and r to (147, 147) and (147, 1).
        Handles rectangular matrices (obs_dim != state_dim).
        """
        if r_thin.ndim == 1: r_thin = r_thin.reshape(-1, 1)
        
        obs_dim, state_dim = H_thin.shape
        
        H_padded = np.zeros((self.x_dim, self.x_dim))
        r_padded = np.zeros((self.x_dim, 1))
        
        # Protect against dimensions exceeding 147
        valid_obs = min(obs_dim, self.x_dim)
        valid_state = min(state_dim, self.x_dim)
        
        # Fill data into top-left corner
        H_padded[:valid_obs, :valid_state] = H_thin[:valid_obs, :valid_state]
        r_padded[:valid_obs, :] = r_thin[:valid_obs, :]
        
        return H_padded, r_padded, (valid_obs, valid_state)

    def sync_propagation(self, _, current_msckf_state):
        """
        [Step 1] Sync Propagation
        使用 MSCKF 的积分增量来驱动影子状态，保证两帧之间影子状态也会动。
        """
        if not self.is_initialized:
            return

        if self.state_pred_past is None:
            # First frame initialization
            self.state_pred_past = current_msckf_state.copy()
            self.current_pred_temp = current_msckf_state.copy()
            return

        # --- Propagation Logic ---
        if self.msckf_post_past_for_prop is not None:
            # 1. Position Delta
            delta_pos = current_msckf_state.position - self.msckf_post_past_for_prop.position
            self.shadow_pos += delta_pos
            
            # 2. Velocity Delta
            delta_vel = current_msckf_state.velocity - self.msckf_post_past_for_prop.velocity
            self.shadow_vel += delta_vel
            
            # 3. Orientation Delta
            # q_curr = q_delta * q_prev  =>  q_delta = q_curr * q_prev^(-1)
            q_prev_inv = np.array([-self.msckf_post_past_for_prop.orientation[0], 
                                   -self.msckf_post_past_for_prop.orientation[1],
                                   -self.msckf_post_past_for_prop.orientation[2],
                                    self.msckf_post_past_for_prop.orientation[3]])
            q_delta = quaternion_multiplication(current_msckf_state.orientation, q_prev_inv)
            self.shadow_q = quaternion_multiplication(q_delta, self.shadow_q)

            # Bias is assumed constant during propagation (Random Walk)
        
        # Store current prediction for feature calculation in Step 2
        self.current_pred_temp = current_msckf_state.copy()

    def get_optimal_gain(self, H_thin, r_thin):
        """
        [Step 2] Core Inference: Calculate Kalman Gain (K) via Neural Network
        """
        if self.kf_net is None: return None

        # 1. Padding & Dimension Check
        H_pad, r_pad, (valid_obs, valid_state) = self._pad_qr_data(H_thin, r_thin)
        
        # [Safety] If observation or state is too small (e.g., < IMU dim), skip
        if valid_obs == 0 or valid_state < 15:
            # print(f"[SKNetAdapter] Info: Small dims (obs={valid_obs}, state={valid_state}), skipping NN.")
            return None

        # 2. History Initialization
        if self.first_run:
            self.obs_past = np.zeros_like(r_pad)
            if self.current_pred_temp is None:
                return None
            self.state_post_past = self.current_pred_temp.copy()
            self.first_run = False

        # 3. Feature Construction
        # Convert all state objects to flattened vectors first
        vec_post_past = self.flatten_msckf_state(self.state_post_past)
        vec_pred_past = self.flatten_msckf_state(self.state_pred_past)
        vec_curr_pred = self.flatten_msckf_state(self.current_pred_temp)

        state_inno = vec_post_past - vec_pred_past
        diff_state = vec_curr_pred - vec_post_past
        diff_obs = r_pad - self.obs_past
        lin_error = r_pad # Approximation

        # 4. To Tensor (With Batch Dim)
        K_valid = None

        try:
            with torch.no_grad():
                def to_tensor(x):
                    return torch.from_numpy(x).float().to(self.device).view(1, -1)

                t_H_flat = to_tensor(H_pad.flatten())
                t_state_inno = to_tensor(state_inno)
                t_r_pad = to_tensor(r_pad)
                t_diff_state = to_tensor(diff_state)
                t_diff_obs = to_tensor(diff_obs)
                t_lin_error = to_tensor(lin_error)
                
                # Check for NaNs
                if torch.isnan(t_state_inno).any() or torch.isnan(t_r_pad).any():
                    print("[SKNetAdapter] Warning: NaN detected in input tensors.")
                    return None

                # Inference
                (Pk, Sk) = self.kf_net(
                    t_state_inno, 
                    t_r_pad, 
                    t_diff_state, 
                    t_diff_obs, 
                    t_lin_error, 
                    t_H_flat
                )
                
                # Recover K = P * H^T * S
                Pk_mat = Pk.view(self.x_dim, self.x_dim) 
                Sk_mat = Sk.view(self.y_dim, self.y_dim)
                t_H_mat = torch.from_numpy(H_pad).float().to(self.device)
                
                K_tensor = Pk_mat @ t_H_mat.T @ Sk_mat
                K_full = K_tensor.cpu().numpy()
                
                # Extract valid sub-matrix
                # K theoretical shape: (active_state_dim, active_obs_dim)
                K_valid = K_full[:valid_state, :valid_obs]
                
        except Exception as e:
            print("\n" + "="*30)
            print(f"[SKNetAdapter] INFERENCE EXCEPTION: {e}")
            print(f"Shapes -> state_inno: {state_inno.shape}, r: {r_pad.shape}")
            traceback.print_exc()
            print("="*30 + "\n")
            return None

        # 5. Update History
        self.obs_past = r_pad.copy()
        self.state_pred_past = self.current_pred_temp.copy()
        
        return K_valid

    def update_shadow_trajectory(self, H, r):
        """
        [Step 3] Update the Shadow State using NN-predicted Gain
        """
        if not self.is_initialized:
            return None

        # Get Predicted Gain
        K_pred = self.get_optimal_gain(H, r)

        if K_pred is None:
            # If inference failed (e.g. device error or dimension error), 
            # return None so vio.py can stop the process strictly.
            return None

        # Calculate Correction: dx = K * r
        # Shape: (valid_state, valid_obs) @ (valid_obs, 1) -> (valid_state, 1)
        delta_x = K_pred @ r 
        dx = delta_x.flatten()

        # [Safety Check] Ensure dx covers at least the IMU state (indices 0-14)
        if len(dx) < 15:
            print(f"[SKNetAdapter] Error: Computed delta_x is too small ({len(dx)}). Need at least 15.")
            return None

        # --- Apply Correction to Shadow State ---
        try:
            # 1. Orientation
            dq = small_angle_quaternion(dx[0:3])
            self.shadow_q = quaternion_multiplication(dq, self.shadow_q)
            
            # 2. Gyro Bias
            self.shadow_bg += dx[3:6]
            
            # 3. Velocity
            self.shadow_vel += dx[6:9]
            
            # 4. Acc Bias
            self.shadow_ba += dx[9:12]
            
            # 5. Position
            self.shadow_pos += dx[12:15]
            
        except Exception as e:
            print(f"[SKNetAdapter] Error applying update to shadow state: {e}")
            return None

        return self.shadow_pos.copy()

    def on_update_finished_msckf(self, state_server):
        """[Step 4] Record the MSCKF Posterior (Target for training next step)"""
        # Save flattened vector for Feature Calc
        self.state_post_past = self.flatten_msckf_state(state_server)
        
        # Save Deep Copy of State Object for Propagation Delta Calc
        self.msckf_post_past_for_prop = copy.deepcopy(state_server.imu_state)