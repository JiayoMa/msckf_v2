import numpy as np
from scipy.stats import chi2

from utils import *
from feature import Feature

import time
from collections import namedtuple
import os
from sknet_adapter import SKNetAdapter 

class IMUState(object):
    # id for next IMU state
    next_id = 0

    # Gravity vector in the world frame
    gravity = np.array([0., 0., -9.81])

    # Transformation offset from the IMU frame to the body frame. 
    # The transformation takes a vector from the IMU frame to the 
    # body frame. The z axis of the body frame should point upwards.
    # Normally, this transform should be identity.
    T_imu_body = Isometry3d(np.identity(3), np.zeros(3))

    def __init__(self, new_id=None):
        # An unique identifier for the IMU state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the IMU (body) frame.
        self.orientation = np.array([0., 0., 0., 1.])

        # Position of the IMU (body) frame in the world frame.
        self.position = np.zeros(3)
        # Velocity of the IMU (body) frame in the world frame.
        self.velocity = np.zeros(3)

        # Bias for measured angular velocity and acceleration.
        self.gyro_bias = np.zeros(3)
        self.acc_bias = np.zeros(3)

        # These three variables should have the same physical
        # interpretation with `orientation`, `position`, and
        # `velocity`. There three variables are used to modify
        # the transition matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros(3)
        self.velocity_null = np.zeros(3)

        # Transformation between the IMU and the left camera (cam0)
        self.R_imu_cam0 = np.identity(3)
        self.t_cam0_imu = np.zeros(3)


class CAMState(object):
    # Takes a vector from the cam0 frame to the cam1 frame.
    R_cam0_cam1 = None
    t_cam0_cam1 = None

    def __init__(self, new_id=None):
        # An unique identifier for the CAM state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the camera frame.
        self.orientation = np.array([0., 0., 0., 1.])

        # Position of the camera frame in the world frame.
        self.position = np.zeros(3)

        # These two variables should have the same physical
        # interpretation with `orientation` and `position`.
        # There two variables are used to modify the measurement
        # Jacobian matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros(3)

        

class StateServer(object):
    def __init__(self, config=None): # 建议传入config以获取max_cam_state_size
        self.imu_state = IMUState()
        self.cam_states = dict()

        # --- 修改开始 ---
        # 假设 config.max_cam_state_size = 20
        # 最大维度 = 21 (IMU) + 6 * 20 (Camera) = 141
        self.max_dim = 147 

        
        # 初始化一个固定的巨大矩阵，全部填0
        self.state_cov = np.zeros((self.max_dim, self.max_dim)) 
        self.continuous_noise_cov = np.zeros((12, 12))
        # --- 修改结束 ---
    
    # 建议添加一个辅助属性方便获取当前有效维度
    @property
    def active_dim(self):
        return 21 + 6 * len(self.cam_states)



class MSCKF(object):
    def __init__(self, config, update_mode="msckf"):
        self.config = config
        self.optimization_config = config.optimization_config
        
        # --- [修改 1] 模式控制 ---
        # mode: 'train' (采集数据) / 'test' (对比实验) / 'normal' (纯MSCKF)
        self.mode = 'test' 
        
        # Update mode for measurement update: "msckf" or "sknet"
        self.update_mode = update_mode
        
        # 初始化 Adapter (采集数据时 model_path 可为 None, device 建议用 cpu 避免显存占用)
        # 如果是 'test' 模式，请确保 model_path 指向真实模型，device='cuda'
        self.sknet_adapter = SKNetAdapter(config, model_path=None, device='cuda')
        
        # IMU data buffer
        self.imu_msg_buffer = []
        
        # --- [修改 2] 数据采集容器 ---
        if self.mode == 'train':
            self.train_data_save_dir = "./train_data_qr"
            os.makedirs(self.train_data_save_dir, exist_ok=True)
            self.data_buffer = []

        # State vector
        self.state_server = StateServer()
        # Features used
        self.map_server = dict()   # <FeatureID, Feature>

        # Chi squared test table.
        self.chi_squared_test_table = dict()
        for i in range(1, 100):
            self.chi_squared_test_table[i] = chi2.ppf(0.05, i)

        # Set the initial IMU state.
        self.state_server.imu_state.velocity = config.velocity
        self.reset_state_cov()

        # Noise Covariance
        continuous_noise_cov = np.identity(12)
        continuous_noise_cov[:3, :3] *= self.config.gyro_noise
        continuous_noise_cov[3:6, 3:6] *= self.config.gyro_bias_noise
        continuous_noise_cov[6:9, 6:9] *= self.config.acc_noise
        continuous_noise_cov[9:, 9:] *= self.config.acc_bias_noise
        self.state_server.continuous_noise_cov = continuous_noise_cov

        # Gravity & Extrinsics
        IMUState.gravity = config.gravity
        T_cam0_imu = np.linalg.inv(config.T_imu_cam0)
        self.state_server.imu_state.R_imu_cam0 = T_cam0_imu[:3, :3].T
        self.state_server.imu_state.t_cam0_imu = T_cam0_imu[:3, 3]

        T_cam0_cam1 = config.T_cn_cnm1
        CAMState.R_cam0_cam1 = T_cam0_cam1[:3, :3]
        CAMState.t_cam0_cam1 = T_cam0_cam1[:3, 3]
        Feature.R_cam0_cam1 = CAMState.R_cam0_cam1
        Feature.t_cam0_cam1 = CAMState.t_cam0_cam1
        IMUState.T_imu_body = Isometry3d(
            config.T_imu_body[:3, :3],
            config.T_imu_body[:3, 3])

        self.tracking_rate = None
        self.is_gravity_set = False
        self.is_first_img = True



    def imu_callback(self, imu_msg):
        """
        Callback function for the imu message.
        """
        # IMU msgs are pushed backed into a buffer instead of being processed 
        # immediately. The IMU msgs are processed when the next image is  
        # available, in which way, we can easily handle the transfer delay.
        self.imu_msg_buffer.append(imu_msg)

        if not self.is_gravity_set:
            if len(self.imu_msg_buffer) >= 200:
                self.initialize_gravity_and_bias()
                self.is_gravity_set = True

    def feature_callback(self, feature_msg):
        """
        Callback function for feature measurements.
        """
        if not self.is_gravity_set:
            return
        start = time.time()

        # Start the system if the first image is received.
        # The frame where the first image is received will be the origin.
        if self.is_first_img:
            self.is_first_img = False
            self.state_server.imu_state.timestamp = feature_msg.timestamp

        t = time.time()

        # Propogate the IMU state.
        # that are received before the image msg.
        self.batch_imu_processing(feature_msg.timestamp)

        print('---batch_imu_processing    ', time.time() - t)
        t = time.time()

        # Augment the state vector.
        self.state_augmentation(feature_msg.timestamp)

        print('---state_augmentation      ', time.time() - t)
        t = time.time()

        # Add new observations for existing features or new features 
        # in the map server.
        self.add_feature_observations(feature_msg)

        print('---add_feature_observations', time.time() - t)
        t = time.time()

        # Perform measurement update if necessary.
        # And prune features and camera states.
        self.remove_lost_features()

        print('---remove_lost_features    ', time.time() - t)
        t = time.time()

        self.prune_cam_state_buffer()

        print('---prune_cam_state_buffer  ', time.time() - t)
        print('---msckf elapsed:          ', time.time() - start, f'({feature_msg.timestamp})')

        try:
            # Publish the odometry.
            return self.publish(feature_msg.timestamp)
        finally:
            # Reset the system if necessary.
            self.online_reset()
    def initialize_gravity_and_bias(self):
        """
        Initialize the IMU bias and initial orientation based on the 
        first few IMU readings.
        """
        sum_angular_vel = np.zeros(3)
        sum_linear_acc = np.zeros(3)
        for msg in self.imu_msg_buffer:
            sum_angular_vel += msg.angular_velocity
            sum_linear_acc += msg.linear_acceleration

        gyro_bias = sum_angular_vel / len(self.imu_msg_buffer)
        self.state_server.imu_state.gyro_bias = gyro_bias

        # This is the gravity in the IMU frame.
        gravity_imu = sum_linear_acc / len(self.imu_msg_buffer)

        # Initialize the initial orientation, so that the estimation
        # is consistent with the inertial frame.
        gravity_norm = np.linalg.norm(gravity_imu)
        IMUState.gravity = np.array([0., 0., -gravity_norm])

        self.state_server.imu_state.orientation = from_two_vectors(
            -IMUState.gravity, gravity_imu)
            
        # ==========================================================
        # [新增/修改] 初始化 SKNet Adapter (关键修复)
        # 必须在算出 orientation 和 gyro_bias 后立即同步给 SKNet
        # ==========================================================
        if self.mode == 'test' and hasattr(self, 'sknet_adapter') and self.sknet_adapter is not None:
            print("[MSCKF] Gravity initialized. Syncing SKNet Shadow State...")
            # 将当前算好的初始状态传给 Adapter
            self.sknet_adapter.initialize_shadow_state(self.state_server.imu_state)
        # ==========================================================
    # Filter related functions
    # (batch_imu_processing, process_model, predict_new_state)
    def batch_imu_processing(self, time_bound):
        """
        Propogate the state
        """
        prev_state_vec = self.sknet_adapter.flatten_msckf_state(self.state_server)
        used_imu_msg_count = 0
        for msg in self.imu_msg_buffer:
            imu_time = msg.timestamp
            if imu_time < self.state_server.imu_state.timestamp:
                used_imu_msg_count += 1
                continue
            if imu_time > time_bound:
                break

            # Execute process model.
            self.process_model(
                imu_time, msg.angular_velocity, msg.linear_acceleration)
            used_imu_msg_count += 1

            # Update the state info
            self.state_server.imu_state.timestamp = imu_time

        self.state_server.imu_state.id = IMUState.next_id
        IMUState.next_id += 1

        # Remove all used IMU msgs.
        self.imu_msg_buffer = self.imu_msg_buffer[used_imu_msg_count:]
                # 2. [SKNet] 传播后同步增量
        curr_state_vec = self.sknet_adapter.flatten_msckf_state(self.state_server)
        self.sknet_adapter.sync_propagation(prev_state_vec, curr_state_vec)

    def process_model(self, time, m_gyro, m_acc):
        imu_state = self.state_server.imu_state
        dt = time - imu_state.timestamp

        gyro = m_gyro - imu_state.gyro_bias
        acc = m_acc - imu_state.acc_bias

        # Compute discrete transition and noise covariance matrix
        F = np.zeros((21, 21))
        G = np.zeros((21, 12))

        R_w_i = to_rotation(imu_state.orientation)

        F[:3, :3] = -skew(gyro)
        F[:3, 3:6] = -np.identity(3)
        F[6:9, :3] = -R_w_i.T @ skew(acc)
        F[6:9, 9:12] = -R_w_i.T
        F[12:15, 6:9] = np.identity(3)

        G[:3, :3] = -np.identity(3)
        G[3:6, 3:6] = np.identity(3)
        G[6:9, 6:9] = -R_w_i.T
        G[9:12, 9:12] = np.identity(3)

        # Approximate matrix exponential to the 3rd order, which can be 
        # considered to be accurate enough assuming dt is within 0.01s.
        Fdt = F * dt
        Fdt_square = Fdt @ Fdt
        Fdt_cube = Fdt_square @ Fdt
        Phi = np.identity(21) + Fdt + Fdt_square/2. + Fdt_cube/6.

        # Propogate the state using 4th order Runge-Kutta
        self.predict_new_state(dt, gyro, acc)

        # Modify the transition matrix
        R_kk_1 = to_rotation(imu_state.orientation_null)
        Phi[:3, :3] = to_rotation(imu_state.orientation) @ R_kk_1.T

        u = R_kk_1 @ IMUState.gravity
        # s = (u.T @ u).inverse() @ u.T
        # s = np.linalg.inv(u[:, None] * u) @ u
        s = u / (u @ u)

        A1 = Phi[6:9, :3]
        w1 = skew(imu_state.velocity_null - imu_state.velocity) @ IMUState.gravity
        Phi[6:9, :3] = A1 - (A1 @ u - w1)[:, None] * s

        A2 = Phi[12:15, :3]
        w2 = skew(dt*imu_state.velocity_null+imu_state.position_null - 
            imu_state.position) @ IMUState.gravity
        Phi[12:15, :3] = A2 - (A2 @ u - w2)[:, None] * s

        curr_dim = self.state_server.active_dim# get current active dimension

        # Propogate the state covariance matrix.
        Q = Phi @ G @ self.state_server.continuous_noise_cov @ G.T @ Phi.T * dt
        self.state_server.state_cov[:21, :21] = (
            Phi @ self.state_server.state_cov[:21, :21] @ Phi.T + Q)
        if len(self.state_server.cam_states) > 0:
            # 只取 21:curr_dim 这一段有效数据
            self.state_server.state_cov[:21, 21:curr_dim] = (
                Phi @ self.state_server.state_cov[:21, 21:curr_dim])
            
            self.state_server.state_cov[21:curr_dim, :21] = (
                self.state_server.state_cov[21:curr_dim, :21] @ Phi.T)



        self.state_server.state_cov[:21, :21] = (
            self.state_server.state_cov[:21, :21] + self.state_server.state_cov[:21, :21].T) / 2.

        # Update the state correspondes to null space.
        self.state_server.imu_state.orientation_null = imu_state.orientation
        self.state_server.imu_state.position_null = imu_state.position
        self.state_server.imu_state.velocity_null = imu_state.velocity

    def predict_new_state(self, dt, gyro, acc):
        # TODO: Will performing the forward integration using
        # the inverse of the quaternion give better accuracy?
        gyro_norm = np.linalg.norm(gyro)
        Omega = np.zeros((4, 4))
        Omega[:3, :3] = -skew(gyro)
        Omega[:3, 3] = gyro
        Omega[3, :3] = -gyro

        q = self.state_server.imu_state.orientation
        v = self.state_server.imu_state.velocity
        p = self.state_server.imu_state.position

        if gyro_norm > 1e-5:
            dq_dt = (np.cos(gyro_norm*dt*0.5) * np.identity(4) + 
                np.sin(gyro_norm*dt*0.5)/gyro_norm * Omega) @ q
            dq_dt2 = (np.cos(gyro_norm*dt*0.25) * np.identity(4) + 
                np.sin(gyro_norm*dt*0.25)/gyro_norm * Omega) @ q
        else:
            dq_dt = np.cos(gyro_norm*dt*0.5) * (np.identity(4) + 
                Omega*dt*0.5) @ q
            dq_dt2 = np.cos(gyro_norm*dt*0.25) * (np.identity(4) + 
                Omega*dt*0.25) @ q

        dR_dt_transpose = to_rotation(dq_dt).T
        dR_dt2_transpose = to_rotation(dq_dt2).T

        # k1 = f(tn, yn)
        k1_p_dot = v
        k1_v_dot = to_rotation(q).T @ acc + IMUState.gravity

        # k2 = f(tn+dt/2, yn+k1*dt/2)
        k1_v = v + k1_v_dot*dt/2.
        k2_p_dot = k1_v
        k2_v_dot = dR_dt2_transpose @ acc + IMUState.gravity
        
        # k3 = f(tn+dt/2, yn+k2*dt/2)
        k2_v = v + k2_v_dot*dt/2
        k3_p_dot = k2_v
        k3_v_dot = dR_dt2_transpose @ acc + IMUState.gravity
        
        # k4 = f(tn+dt, yn+k3*dt)
        k3_v = v + k3_v_dot*dt
        k4_p_dot = k3_v
        k4_v_dot = dR_dt_transpose @ acc + IMUState.gravity

        # yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
        q = dq_dt / np.linalg.norm(dq_dt)
        v = v + (k1_v_dot + 2*k2_v_dot + 2*k3_v_dot + k4_v_dot)*dt/6.
        p = p + (k1_p_dot + 2*k2_p_dot + 2*k3_p_dot + k4_p_dot)*dt/6.

        self.state_server.imu_state.orientation = q
        self.state_server.imu_state.velocity = v
        self.state_server.imu_state.position = p
    def state_augmentation(self, time):
        imu_state = self.state_server.imu_state
        R_i_c = imu_state.R_imu_cam0
        t_c_i = imu_state.t_cam0_imu

        # 1. 这一步已经把新相机加入了字典，导致 len(cam_states) 增加了
        R_w_i = to_rotation(imu_state.orientation)
        R_w_c = R_i_c @ R_w_i
        t_c_w = imu_state.position + R_w_i.T @ t_c_i

        cam_state = CAMState(imu_state.id)
        cam_state.timestamp = time
        cam_state.orientation = to_quaternion(R_w_c)
        cam_state.position = t_c_w
        cam_state.orientation_null = cam_state.orientation
        cam_state.position_null = cam_state.position
        self.state_server.cam_states[imu_state.id] = cam_state

        # ... (J 矩阵计算代码保持不变) ...
        J = np.zeros((6, 21))
        J[:3, :3] = R_i_c
        J[:3, 15:18] = np.identity(3)
        J[3:6, :3] = skew(R_w_i.T @ t_c_i)
        J[3:6, 12:15] = np.identity(3)
        J[3:6, 18:21] = np.identity(3)

        # --- 修正开始 ---
        # 错误代码: old_size = self.state_server.active_dim 
        # 正确代码: 必须减去6，因为字典里已经包含了这个新相机，但协方差矩阵还没填
        old_size = self.state_server.active_dim - 6 

        # 2. 越界检查
        if old_size + 6 > self.state_server.max_dim:
            print("Error: Exceeded max state size")
            # 如果这步失败，必须把刚才加进去的相机从字典里删掉，否则状态就不一致了
            del self.state_server.cam_states[imu_state.id] 
            return

        # 3. 填充协方差
        P = self.state_server.state_cov
        
        # P_new_top_right = P_old * J.T
        P[:old_size, old_size:old_size+6] = P[:old_size, :21] @ J.T
        
        # P_new_bottom_left = J * P_old
        P[old_size:old_size+6, :old_size] = J @ P[:21, :old_size]
        
        # P_new_bottom_right = J * P_imu * J.T
        P[old_size:old_size+6, old_size:old_size+6] = J @ P[:21, :21] @ J.T

        # 强制对称
        active_block = P[:old_size+6, :old_size+6]
        P[:old_size+6, :old_size+6] = (active_block + active_block.T) / 2.
        # --- 修正结束 ---

    def add_feature_observations(self, feature_msg):
        state_id = self.state_server.imu_state.id
        curr_feature_num = len(self.map_server)
        tracked_feature_num = 0

        for feature in feature_msg.features:
            if feature.id not in self.map_server:
                # This is a new feature.
                map_feature = Feature(feature.id, self.optimization_config)
                map_feature.observations[state_id] = np.array([
                    feature.u0, feature.v0, feature.u1, feature.v1])
                self.map_server[feature.id] = map_feature
            else:
                # This is an old feature.
                self.map_server[feature.id].observations[state_id] = np.array([
                    feature.u0, feature.v0, feature.u1, feature.v1])
                tracked_feature_num += 1

        self.tracking_rate = tracked_feature_num / (curr_feature_num+1e-5)

    def measurement_jacobian(self, cam_state_id, feature_id):
        """
        This function is used to compute the measurement Jacobian
        for a single feature observed at a single camera frame.
        """
        # Prepare all the required data.
        cam_state = self.state_server.cam_states[cam_state_id]
        feature = self.map_server[feature_id]

        # Cam0 pose.
        R_w_c0 = to_rotation(cam_state.orientation)
        t_c0_w = cam_state.position

        # Cam1 pose.
        R_w_c1 = CAMState.R_cam0_cam1 @ R_w_c0
        t_c1_w = t_c0_w - R_w_c1.T @ CAMState.t_cam0_cam1

        # 3d feature position in the world frame.
        # And its observation with the stereo cameras.
        p_w = feature.position
        z = feature.observations[cam_state_id]

        # Convert the feature position from the world frame to
        # the cam0 and cam1 frame.
        p_c0 = R_w_c0 @ (p_w - t_c0_w)
        p_c1 = R_w_c1 @ (p_w - t_c1_w)

        # Compute the Jacobians.
        dz_dpc0 = np.zeros((4, 3))
        dz_dpc0[0, 0] = 1 / p_c0[2]
        dz_dpc0[1, 1] = 1 / p_c0[2]
        dz_dpc0[0, 2] = -p_c0[0] / (p_c0[2] * p_c0[2])
        dz_dpc0[1, 2] = -p_c0[1] / (p_c0[2] * p_c0[2])

        dz_dpc1 = np.zeros((4, 3))
        dz_dpc1[2, 0] = 1 / p_c1[2]
        dz_dpc1[3, 1] = 1 / p_c1[2]
        dz_dpc1[2, 2] = -p_c1[0] / (p_c1[2] * p_c1[2])
        dz_dpc1[3, 2] = -p_c1[1] / (p_c1[2] * p_c1[2])

        dpc0_dxc = np.zeros((3, 6))
        dpc0_dxc[:, :3] = skew(p_c0)
        dpc0_dxc[:, 3:] = -R_w_c0

        dpc1_dxc = np.zeros((3, 6))
        dpc1_dxc[:, :3] = CAMState.R_cam0_cam1 @ skew(p_c0)
        dpc1_dxc[:, 3:] = -R_w_c1

        dpc0_dpg = R_w_c0
        dpc1_dpg = R_w_c1

        H_x = dz_dpc0 @ dpc0_dxc + dz_dpc1 @ dpc1_dxc   # shape: (4, 6)
        H_f = dz_dpc0 @ dpc0_dpg + dz_dpc1 @ dpc1_dpg   # shape: (4, 3)

        # Modifty the measurement Jacobian to ensure observability constrain.
        A = H_x   # shape: (4, 6)
        u = np.zeros(6)
        u[:3] = to_rotation(cam_state.orientation_null) @ IMUState.gravity
        u[3:] = skew(p_w - cam_state.position_null) @ IMUState.gravity

        H_x = A - (A @ u)[:, None] * u / (u @ u)
        H_f = -H_x[:4, 3:6]

        # Compute the residual.
        r = z - np.array([*p_c0[:2]/p_c0[2], *p_c1[:2]/p_c1[2]])

        # H_x: shape (4, 6)
        # H_f: shape (4, 3)
        # r  : shape (4,)
        return H_x, H_f, r

    def feature_jacobian(self, feature_id, cam_state_ids):
        """
        This function computes the Jacobian of all measurements viewed 
        in the given camera states of this feature.
        """
        feature = self.map_server[feature_id]

        # Check how many camera states in the provided camera id 
        # camera has actually seen this feature.
        valid_cam_state_ids = []
        for cam_id in cam_state_ids:
            if cam_id in feature.observations:
                valid_cam_state_ids.append(cam_id)

        jacobian_row_size = 4 * len(valid_cam_state_ids)

        cam_states = self.state_server.cam_states
        H_xj = np.zeros((jacobian_row_size, 
            21+len(self.state_server.cam_states)*6))
        H_fj = np.zeros((jacobian_row_size, 3))
        r_j = np.zeros(jacobian_row_size)

        stack_count = 0
        for cam_id in valid_cam_state_ids:
            H_xi, H_fi, r_i = self.measurement_jacobian(cam_id, feature.id)

            # Stack the Jacobians.
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            H_xj[stack_count:stack_count+4, 21+6*idx:21+6*(idx+1)] = H_xi
            H_fj[stack_count:stack_count+4, :3] = H_fi
            r_j[stack_count:stack_count+4] = r_i
            stack_count += 4

        # Project the residual and Jacobians onto the nullspace of H_fj.
        # svd of H_fj
        U, _, _ = np.linalg.svd(H_fj)
        A = U[:, 3:]

        H_x = A.T @ H_xj
        r = A.T @ r_j

        return H_x, r
    def measurement_update(self, H, r, update_mode="msckf"):
        """
        MSCKF Measurement Update with optional SKNet integration
        
        Args:
            H: Observation matrix
            r: Residual vector
            update_mode: "msckf" (traditional update) or "sknet" (use SKNet's K, Pk, Sk)
        
        Returns:
            For backward compatibility with existing code
        """
        if len(H) == 0 or len(r) == 0:
            return None

        # 1. QR Decomposition
        if H.shape[0] > H.shape[1]:
            Q, R_qr = np.linalg.qr(H, mode='reduced')
            H_thin = R_qr       
            r_thin = Q.T @ r   
        else:
            H_thin = H   
            r_thin = r   

        # --- [修改 3] 关键：获取当前先验状态 (State Prediction) ---
        # 此时 state_server 尚未更新，所以这就是 x_{k|k-1}
        # 如果不保存这个，离线训练时就无法计算 diff_state
        state_pred_vec = self.sknet_adapter.flatten_msckf_state(self.state_server)

        # --- [修改 4] 分支 A：训练模式 (保存数据) ---
        if self.mode == 'train':
            self.data_buffer.append({
                'H': H_thin,
                'r': r_thin,
                'timestamp': self.state_server.imu_state.timestamp,
                'state_pred': state_pred_vec  # <--- 必须保存这个！
            })
            # 每 500 帧存一次，防止内存爆炸
            if len(self.data_buffer) >= 500: 
                self.flush_training_data()

        # --- Get active dimension and covariance ---
        curr_dim = self.state_server.active_dim
        P_active = self.state_server.state_cov[:curr_dim, :curr_dim]
        
        # --- Branch based on update_mode ---
        if update_mode == "sknet":
            # Use SKNet's predicted Kalman gain and covariances
            result = self.sknet_adapter.get_optimal_gain(H_thin, r_thin, return_covariances=True)
            
            if result is None or result[0] is None:
                print("\033[91m[Critical] SKNet inference failed. Cannot update state.\033[0m")
                raise RuntimeError("SKNet inference failed.")
            
            K, Pk_sknet, Sk_sknet = result
            
            # For covariance update, we use SKNet's Pk
            # Note: Pk_sknet should match curr_dim, so extract the active portion
            if Pk_sknet.shape[0] < curr_dim or Pk_sknet.shape[1] < curr_dim:
                print(f"[Warning] SKNet Pk dimension ({Pk_sknet.shape}) < active dim ({curr_dim})")
                # Fallback to traditional update
                update_mode = "msckf"
            else:
                P_active = Pk_sknet[:curr_dim, :curr_dim]
        
        if update_mode == "msckf":
            # Traditional MSCKF update: Calculate S and K from P
            S = H_thin @ P_active @ H_thin.T + (self.config.observation_noise * np.identity(len(H_thin)))
            K_transpose = np.linalg.solve(S, H_thin @ P_active)
            K = K_transpose.T

        # Calculate delta_x using the appropriate K
        delta_x = K @ r_thin 
        
        # Update the IMU state
        delta_x_imu = delta_x[:21]
        if (np.linalg.norm(delta_x_imu[6:9]) > 0.5 or 
            np.linalg.norm(delta_x_imu[12:15]) > 1.0):
            print('[Warning] Update change is too large')

        dq_imu = small_angle_quaternion(delta_x_imu[:3])
        imu_state = self.state_server.imu_state
        imu_state.orientation = quaternion_multiplication(dq_imu, imu_state.orientation)
        imu_state.gyro_bias += delta_x_imu[3:6]
        imu_state.velocity += delta_x_imu[6:9]
        imu_state.acc_bias += delta_x_imu[9:12]
        imu_state.position += delta_x_imu[12:15]

        dq_extrinsic = small_angle_quaternion(delta_x_imu[15:18])
        imu_state.R_imu_cam0 = to_rotation(dq_extrinsic) @ imu_state.R_imu_cam0
        imu_state.t_cam0_imu += delta_x_imu[18:21]
        
        # Update camera states
        for i, (cam_id, cam_state) in enumerate(self.state_server.cam_states.items()):
            delta_x_cam = delta_x[21+i*6:27+i*6]
            dq_cam = small_angle_quaternion(delta_x_cam[:3])
            cam_state.orientation = quaternion_multiplication(dq_cam, cam_state.orientation)
            cam_state.position += delta_x_cam[3:]

        # Update covariance
        if update_mode == "sknet":
            # Use SKNet's posterior covariance directly
            # Apply Joseph form for numerical stability: P_post = (I - KH)P_pred(I - KH)^T + KRK^T
            # For simplicity, we use the provided Pk as the posterior
            self.state_server.state_cov[:curr_dim, :curr_dim] = P_active
        else:
            # Traditional MSCKF covariance update
            I_KH = np.identity(curr_dim) - K @ H_thin
            P_new_active = I_KH @ P_active
            self.state_server.state_cov[:curr_dim, :curr_dim] = (P_new_active + P_new_active.T) / 2.
        
        # --- [修改 6] 通知 Adapter 更新完成 (用于计算 state_inno) ---
        self.sknet_adapter.on_update_finished_msckf(self.state_server)
        
        return None  # Return None for consistency
        
    def flush_training_data(self):
        """将数据保存为 .npy 文件"""
        if not self.data_buffer:
            return
            
        # 自动编号 batch_0.npy, batch_1.npy ...
        idx = len(os.listdir(self.train_data_save_dir))
        save_path = os.path.join(self.train_data_save_dir, f"batch_{idx}.npy")
        
        print(f"Saving training batch {idx} with {len(self.data_buffer)} samples...")
        np.save(save_path, self.data_buffer)
        
        # 清空 buffer 释放内存
        self.data_buffer = []

    # 别忘了在程序结束时 (例如 vio.py 退出时) 再次调用 flush，否则最后一部分数据会丢失
    def finalize(self):
        if self.mode == 'train':
            self.flush_training_data()
    def gating_test(self, H, r, dof):
        # --- 修改开始 ---
        # 1. 获取当前有效维度
        curr_dim = self.state_server.active_dim
        
        # 2. 从大矩阵中切片出有效的协方差矩阵 (Small Matrix)
        # 这样 P_active 的维度就是 (63, 63)，与 H 的列数 (63) 匹配
        P_active = self.state_server.state_cov[:curr_dim, :curr_dim]
        
        # 3. 使用切片后的 P_active 进行计算
        P1 = H @ P_active @ H.T
        # --- 修改结束 ---
        
        P2 = self.config.observation_noise * np.identity(len(H))
        gamma = r @ np.linalg.solve(P1+P2, r)

        if(gamma < self.chi_squared_test_table[dof]):
            return True
        else:
            return False

    def remove_lost_features(self):
        # Remove the features that lost track.
        # BTW, find the size the final Jacobian matrix and residual vector.
        jacobian_row_size = 0
        invalid_feature_ids = []
        processed_feature_ids = []

        for feature in self.map_server.values():
            # Pass the features that are still being tracked.
            if self.state_server.imu_state.id in feature.observations:
                continue
            if len(feature.observations) < 3:
                invalid_feature_ids.append(feature.id)
                continue

            # Check if the feature can be initialized if it has not been.
            if not feature.is_initialized:
                # Ensure there is enough translation to triangulate the feature
                if not feature.check_motion(self.state_server.cam_states):
                    invalid_feature_ids.append(feature.id)
                    continue

                # Intialize the feature position based on all current available 
                # measurements.
                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    invalid_feature_ids.append(feature.id)
                    continue

            jacobian_row_size += (4 * len(feature.observations) - 3)
            processed_feature_ids.append(feature.id)

        # Remove the features that do not have enough measurements.
        for feature_id in invalid_feature_ids:
            del self.map_server[feature_id]

        # Return if there is no lost feature to be processed.
        if len(processed_feature_ids) == 0:
            return

        H_x = np.zeros((jacobian_row_size, 
            21+6*len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)
        stack_count = 0

        # Process the features which lose track.
        for feature_id in processed_feature_ids:
            feature = self.map_server[feature_id]

            cam_state_ids = []
            for cam_id, measurement in feature.observations.items():
                cam_state_ids.append(cam_id)

            H_xj, r_j = self.feature_jacobian(feature.id, cam_state_ids)

            if self.gating_test(H_xj, r_j, len(cam_state_ids)-1):
                H_x[stack_count:stack_count+H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count+len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            # Put an upper bound on the row size of measurement Jacobian,
            # which helps guarantee the executation time.
            if stack_count > 1500:
                break

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform the measurement update step.
        self.measurement_update(H_x, r, update_mode=self.update_mode)

        # Remove all processed features from the map.
        for feature_id in processed_feature_ids:
            del self.map_server[feature_id]

    def find_redundant_cam_states(self):
        # Move the iterator to the key position.
        cam_state_pairs = list(self.state_server.cam_states.items())

        key_cam_state_idx = len(cam_state_pairs) - 4
        cam_state_idx = key_cam_state_idx + 1
        first_cam_state_idx = 0

        # Pose of the key camera state.
        key_position = cam_state_pairs[key_cam_state_idx][1].position
        key_rotation = to_rotation(
            cam_state_pairs[key_cam_state_idx][1].orientation)

        rm_cam_state_ids = []

        # Mark the camera states to be removed based on the
        # motion between states.
        for i in range(2):
            position = cam_state_pairs[cam_state_idx][1].position
            rotation = to_rotation(
                cam_state_pairs[cam_state_idx][1].orientation)
            
            distance = np.linalg.norm(position - key_position)
            angle = 2 * np.arccos(to_quaternion(
                rotation @ key_rotation.T)[-1])

            if angle < 0.2618 and distance < 0.4 and self.tracking_rate > 0.5:
                rm_cam_state_ids.append(cam_state_pairs[cam_state_idx][0])
                cam_state_idx += 1
            else:
                rm_cam_state_ids.append(cam_state_pairs[first_cam_state_idx][0])
                first_cam_state_idx += 1
                cam_state_idx += 1

        # Sort the elements in the output list.
        rm_cam_state_ids = sorted(rm_cam_state_ids)
        return rm_cam_state_ids


    def prune_cam_state_buffer(self):
        if len(self.state_server.cam_states) < self.config.max_cam_state_size:
            return

        # Find two camera states to be removed.
        rm_cam_state_ids = self.find_redundant_cam_states()

        # Find the size of the Jacobian matrix.
        jacobian_row_size = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue
            if len(involved_cam_state_ids) == 1:
                del feature.observations[involved_cam_state_ids[0]]
                continue

            if not feature.is_initialized:
                # Check if the feature can be initialize.
                if not feature.check_motion(self.state_server.cam_states):
                    # If the feature cannot be initialized, just remove
                    # the observations associated with the camera states
                    # to be removed.
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

            jacobian_row_size += 4*len(involved_cam_state_ids) - 3

        # Compute the Jacobian and residual.
        H_x = np.zeros((jacobian_row_size, 21+6*len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)

        stack_count = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue

            H_xj, r_j = self.feature_jacobian(feature.id, involved_cam_state_ids)

            if self.gating_test(H_xj, r_j, len(involved_cam_state_ids)):
                H_x[stack_count:stack_count+H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count+len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            for cam_id in involved_cam_state_ids:
                del feature.observations[cam_id]

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform measurement update.
        self.measurement_update(H_x, r, update_mode=self.update_mode)
        for cam_id in rm_cam_state_ids:
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            cam_state_start = 21 + 6*idx
            cam_state_end = cam_state_start + 6

            # --- 修改开始 ---
            # 获取当前总有效长度
            current_dim = self.state_server.active_dim 
            
            # 移动数据：将删除块之后的数据块，整体向左上移动
            # 注意：我们需要操作的是 self.state_server.state_cov
            P = self.state_server.state_cov
            
            # 如果被删除的不是最后一个相机，则需要移动数据
            if cam_state_end < current_dim:
                # 1. 行上移 (覆盖掉 cam_state_start 到 cam_state_end 的行)
                P[cam_state_start : current_dim - 6, :] = P[cam_state_end : current_dim, :]
                # 2. 列左移
                P[:, cam_state_start : current_dim - 6] = P[:, cam_state_end : current_dim]
                
                # (可选) 将移动后腾出的末尾部分置零，保持整洁
                P[current_dim-6 : current_dim, :] = 0.0
                P[:, current_dim-6 : current_dim] = 0.0
            del self.state_server.cam_states[cam_id]


    def reset_state_cov(self):
        """
        Reset the state covariance.
        """
        # 1. 不要创建新矩阵，而是获取大矩阵的引用
        P = self.state_server.state_cov
        
        # 2. 将大矩阵全部清零（或者只清零左上角21x21，为了安全建议全清）
        P.fill(0.0)
        
        # 3. 原地设置初始值
        P[ 3: 6,  3: 6] = self.config.gyro_bias_cov * np.identity(3)
        P[ 6: 9,  6: 9] = self.config.velocity_cov * np.identity(3)
        P[ 9:12,  9:12] = self.config.acc_bias_cov * np.identity(3)
        P[15:18, 15:18] = self.config.extrinsic_rotation_cov * np.identity(3)
        P[18:21, 18:21] = self.config.extrinsic_translation_cov * np.identity(3)
        
        # 4. 绝对不要执行下面这行赋值！
        # self.state_server.state_cov = state_cov

    def reset(self):
        """
        Reset the VIO to initial status.
        """
        # Reset the IMU state.
        imu_state = IMUState()
        imu_state.id = self.state_server.imu_state.id
        imu_state.R_imu_cam0 = self.state_server.imu_state.R_imu_cam0
        imu_state.t_cam0_imu = self.state_server.imu_state.t_cam0_imu
        self.state_server.imu_state = imu_state

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Reset the state covariance.
        self.reset_state_cov()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Clear the IMU msg buffer.
        self.imu_msg_buffer.clear()

        # Reset the starting flags.
        self.is_gravity_set = False
        self.is_first_img = True

    def online_reset(self):
        """
        Reset the system online if the uncertainty is too large.
        """
        # Never perform online reset if position std threshold is non-positive.
        if self.config.position_std_threshold <= 0:
            return

        # Check the uncertainty of positions to determine if 
        # the system can be reset.
        position_x_std = np.sqrt(self.state_server.state_cov[12, 12])
        position_y_std = np.sqrt(self.state_server.state_cov[13, 13])
        position_z_std = np.sqrt(self.state_server.state_cov[14, 14])

        if max(position_x_std, position_y_std, position_z_std 
            ) < self.config.position_std_threshold:
            return

        print('Start online reset...')

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Reset the state covariance.
        self.reset_state_cov()

    def publish(self, time):
        imu_state = self.state_server.imu_state
        print('+++publish:')
        print('   timestamp:', imu_state.timestamp)
        print('   orientation:', imu_state.orientation)
        print('   position:', imu_state.position)
        print('   velocity:', imu_state.velocity)
        print()
        
        T_i_w = Isometry3d(
            to_rotation(imu_state.orientation).T,
            imu_state.position)
        T_b_w = IMUState.T_imu_body * T_i_w * IMUState.T_imu_body.inverse()
        body_velocity = IMUState.T_imu_body.R @ imu_state.velocity

        R_w_c = imu_state.R_imu_cam0 @ T_i_w.R.T
        t_c_w = imu_state.position + T_i_w.R @ imu_state.t_cam0_imu
        T_c_w = Isometry3d(R_w_c.T, t_c_w)

        return namedtuple('vio_result', ['timestamp', 'pose', 'velocity', 'cam0_pose'])(
            time, T_b_w, body_velocity, T_c_w)