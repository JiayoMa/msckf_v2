import torch
import torch.nn as nn
import configparser

# 尝试读取配置，如果失败则给默认值，防止报错
try:
    config = configparser.ConfigParser()
    config.read('./config.ini')
    nGRU = int(config['DNN.size']['nGRU'])
    gru_scale_s = int(config['DNN.size']['gru_scale_s']) # 保留读取逻辑但下面会覆盖
    gru_scale_k = int(config['DNN.size']['gru_scale_k'])
except:
    nGRU = 2
    gru_scale_s = 2
    gru_scale_k = 2

class DNN_SKalmanNet_GSS(torch.nn.Module):
    def __init__(self, x_dim:int=2, y_dim:int=2):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # --- [关键配置] 强制设定合理的网络宽度 ---
        # 针对 147 维的高维状态，必须固定宽度，否则参数爆炸
        H1 = 256  # 特征提取层
        H2 = 128  # 解码层
        HIDDEN_DIM = 256 # GRU 隐藏层

        # Input dimensions
        # 对应 forward 中的 input1 拼接: x + dx + lin_err + H_flat
        self.input_dim_1 = (self.x_dim) * 2 + (self.y_dim) + (self.x_dim * self.y_dim)
        # 对应 forward 中的 input2 拼接: y + dy + lin_err + H_flat
        self.input_dim_2 = (self.y_dim) * 2 + (self.y_dim) + (self.x_dim * self.y_dim)

        self.output_dim_1 = (self.x_dim * self.x_dim) 
        self.output_dim_2 = (self.y_dim * self.y_dim)

        # --- Branch 1: State Covariance (Pk) ---
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim_1, H1),
            nn.ReLU()
        )

        self.gru_input_dim = H1
        self.gru_hidden_dim = HIDDEN_DIM
        self.gru_n_layer = nGRU
        self.batch_size = 1 # 初始化默认值

        self.GRU1 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)
        # 初始化隐状态 (Layer, Batch, Hidden)
        self.hn1 = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)

        self.l2 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim_1)
        )

        # --- Branch 2: Innovation Covariance / Gain (Sk) ---
        self.l3 = nn.Sequential(
            nn.Linear(self.input_dim_2, H1),
            nn.ReLU()
        )

        self.GRU2 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)
        self.hn2 = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)     

        self.l4 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim_2)
        )

    def initialize_hidden(self):
        """重置隐状态，通常在推理开始前调用"""
        device = next(self.parameters()).device
        # 默认重置为 batch=1，forward 中会自动调整
        self.hn1 = torch.randn(self.gru_n_layer, 1, self.gru_hidden_dim).to(device)
        self.hn2 = torch.randn(self.gru_n_layer, 1, self.gru_hidden_dim).to(device)

    def forward(self, state_inno, observation_inno, diff_state, diff_obs, linearization_error, Jacobian):
        """
        Args:
            state_inno: (Batch, x_dim)
            observation_inno: (Batch, y_dim) -> residual
            diff_state: (Batch, x_dim)
            diff_obs: (Batch, y_dim)
            linearization_error: (Batch, y_dim)
            Jacobian: (Batch, x_dim*y_dim) -> Flattened H
        """
        

        curr_batch_size = state_inno.shape[0]
        curr_device = state_inno.device # 获取输入数据的设备 (CPU/GPU)
        if (self.hn1.shape[1] != curr_batch_size) or (self.hn1.device != curr_device):
            self.hn1 = torch.randn(self.gru_n_layer, curr_batch_size, self.gru_hidden_dim).to(curr_device)
            self.hn2 = torch.randn(self.gru_n_layer, curr_batch_size, self.gru_hidden_dim).to(curr_device)
        else:
            # 训练时截断梯度流
            self.hn1 = self.hn1.detach()
            self.hn2 = self.hn2.detach()

        # --- [分支 1] Pk 计算 ---
        # 拼接: state(x) + diff_state(x) + lin_error(y) + Jacobian(x*y)
        input1 = torch.cat((state_inno, diff_state, linearization_error, Jacobian), 1)
        
        l1_out = self.l1(input1)
        
        # GRU 输入格式: (Seq_Len=1, Batch, Feature)
        gru1_in = l1_out.unsqueeze(0) 
        gru1_out, self.hn1 = self.GRU1(gru1_in, self.hn1)
        x1 = gru1_out.squeeze(0)
        
        l2_out = self.l2(x1)
        
        # --- [分支 2] Sk 计算 ---
        # 拼接: residual(y) + diff_obs(y) + lin_error(y) + Jacobian(x*y)
        input2 = torch.cat((observation_inno, diff_obs, linearization_error, Jacobian), 1)
        
        l3_out = self.l3(input2)
        
        gru2_in = l3_out.unsqueeze(0)
        gru2_out, self.hn2 = self.GRU2(gru2_in, self.hn2)
        x2 = gru2_out.squeeze(0)
        
        l4_out = self.l4(x2)

        # 输出 (Batch, Dim*Dim)
        Pk = l2_out
        Sk = l4_out
        
        return Pk, Sk

class KNet_architecture_v2(torch.nn.Module):
    def __init__(self, x_dim:int=2, y_dim:int=2, in_mult=20, out_mult=40):
        super().__init__()

        self.gru_num_param_scale = 1

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.gru_n_layer = 1
        self.batch_size = 1
        self.seq_len_input = 1 # Forward 전후로 처리해야할게 있으니 hidden 초기화를 직접 하고 시퀀스 길이를 1로!

        self.prior_Q = torch.eye(x_dim)
        self.prior_Sigma = torch.randn((x_dim,x_dim))
        self.prior_S = torch.randn((y_dim,y_dim))

        # GRU to track Q (5 x 5)
        self.d_input_Q = self.x_dim * in_mult
        self.d_hidden_Q = self.gru_num_param_scale * self.x_dim ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)
        self.h_Q = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_Q)
        # self.h_Q_init = self.h_Q.detach().clone()

        # GRU to track Sigma (5 x 5)
        self.d_input_Sigma = self.d_hidden_Q + self.x_dim * in_mult
        self.d_hidden_Sigma = self.gru_num_param_scale * self.x_dim ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
        self.h_Sigma = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_Sigma)
        # self.h_Sigma_init = self.h_Sigma.detach().clone()

        # GRU to track S (2 x 2)
        self.d_input_S = self.y_dim ** 2 + 2 * self.y_dim * in_mult
        self.d_hidden_S = self.gru_num_param_scale * self.y_dim ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        self.h_S = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_S)
        # self.h_S_init = self.h_S.detach().clone()

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.y_dim ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.y_dim * self.x_dim
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.x_dim ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU())
        
        # Fully connected 5
        self.d_input_FC5 = self.x_dim
        self.d_output_FC5 = self.x_dim * in_mult
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU())

        # Fully connected 6
        self.d_input_FC6 = self.x_dim
        self.d_output_FC6 = self.x_dim * in_mult
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.ReLU())
        
        # Fully connected 7
        self.d_input_FC7 = 2 * self.y_dim
        self.d_output_FC7 = 2 * self.y_dim * in_mult
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU())

    def initialize_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, 0, :] = self.prior_S.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, 0, :] = self.prior_Q.flatten()

    def forward(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff.reshape((-1)))
        obs_innov_diff = expand_dim(obs_innov_diff.reshape((-1)))
        fw_evol_diff = expand_dim(fw_evol_diff.reshape((-1)))
        fw_update_diff = expand_dim(fw_update_diff.reshape((-1)))

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        """
        # FC 8
        in_FC8 = out_Q
        out_FC8 = self.FC8(in_FC8)
        """

        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2.reshape((self.x_dim, self.y_dim))
