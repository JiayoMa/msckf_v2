# SKNet Integration with MSCKF - Implementation Guide

## Overview

This implementation integrates SKNet (Selective Kalman Network) into the MSCKF (Multi-State Constraint Kalman Filter) main state update loop, allowing for a true comparison between traditional MSCKF and SKNet-enhanced MSCKF.

## Key Changes

### 1. Modified Files

#### `sknet_adapter.py`
- **Enhanced `get_optimal_gain()` method**:
  - Added `return_covariances` parameter (default: `False` for backward compatibility)
  - When `return_covariances=True`, returns `(K, Pk, Sk)` tuple
  - When `return_covariances=False`, returns only `K` (original behavior)
  - Extracts valid sub-matrices from padded covariances

#### `msckf.py`
- **MSCKF Constructor**:
  - Added `update_mode` parameter (default: `"msckf"`)
  - Stores `self.update_mode` for use in measurement updates
  
- **measurement_update() method**:
  - Added `update_mode` parameter (default: `"msckf"`)
  - Implements branching logic:
    - `update_mode="msckf"`: Traditional Kalman filter update (P/S/K calculation)
    - `update_mode="sknet"`: Uses SKNet's predicted Pk/Sk for state update
  - Both modes update the main state_server (not shadow trajectory)
  - Updated both call sites to pass `self.update_mode`

#### `vio.py`
- **VIO Constructor**:
  - Added `enable_sknet_comparison` parameter (default: `False`)
  - When enabled, creates two MSCKF instances:
    - `msckf_baseline`: Traditional MSCKF (`update_mode="msckf"`)
    - `msckf_sknet`: SKNet-fusion MSCKF (`update_mode="sknet"`)
  
- **process_imu() method**:
  - Feeds IMU data to both MSCKF instances when dual mode is enabled
  
- **process_feature() method**:
  - Processes features through both MSCKF instances in parallel
  - Collects trajectories from main states (not shadow trajectories)
  - Both trajectories are true closed-loop filter results
  
- **Command-line interface**:
  - Added `--dual_mode` flag to enable comparison mode

## Usage

### Single Mode (Traditional - Backward Compatible)

```bash
python msckfv2/vio.py --path /path/to/dataset --save_dir ./results
```

This runs the system in single mode with the default MSCKF implementation.

### Dual Mode (Comparison)

```bash
python msckfv2/vio.py --path /path/to/dataset --save_dir ./results --dual_mode
```

This runs two MSCKF instances simultaneously:
1. **Baseline MSCKF**: Traditional Kalman filter update
2. **SKNet-Fusion MSCKF**: Uses SKNet's learned Kalman gain

Both trajectories are saved and compared in the output plots.

## Architecture

### Data Flow (Dual Mode)

```
IMU Data → ImageProcessor → Feature Extraction
                ↓
        ┌───────┴────────┐
        ↓                ↓
   MSCKF Baseline   MSCKF SKNet
   (update_mode=   (update_mode=
    "msckf")        "sknet")
        ↓                ↓
   Traditional      SKNet Gain
   P/S/K Calc      (K, Pk, Sk)
        ↓                ↓
   Main State      Main State
   Update          Update
        ↓                ↓
   Trajectory      Trajectory
   (Baseline)      (SKNet)
        └────────┬────────┘
                 ↓
          Comparison Plot
```

### Update Mode Details

#### MSCKF Mode (Traditional)
```python
# Calculate innovation covariance
S = H @ P @ H.T + R

# Calculate Kalman gain
K = P @ H.T @ inv(S)

# Update state
x_post = x_pred + K @ residual

# Update covariance
P_post = (I - K @ H) @ P_pred
```

#### SKNet Mode (Neural Network)
```python
# Get SKNet predictions
(K, Pk, Sk) = sknet_adapter.get_optimal_gain(H, r, return_covariances=True)

# Update state using SKNet's Kalman gain
x_post = x_pred + K @ residual

# Update covariance using SKNet's posterior covariance
P_post = Pk
```

## Important Notes

### 1. Main State vs Shadow Trajectory
- **Previous implementation**: SKNet only updated a "shadow trajectory"
- **Current implementation**: SKNet updates the main MSCKF state
- Both MSCKF instances maintain independent state_servers
- Trajectories are extracted from `state_server.imu_state.position`

### 2. Model Requirements
To use SKNet mode effectively, you need:
- A trained SKNet model (`.pth` file)
- Update the model path in `MSCKF.__init__()`:
  ```python
  self.sknet_adapter = SKNetAdapter(config, model_path="path/to/model.pth", device='cuda')
  ```

### 3. Covariance Handling
- SKNet provides both Pk (posterior covariance) and Sk (innovation covariance)
- The implementation uses Pk directly as the posterior covariance
- Dimension matching is handled automatically (padding/truncation as needed)

### 4. Feature Sharing
- Both MSCKF instances share the same `ImageProcessor`
- Each MSCKF maintains its own:
  - State server (position, velocity, orientation, biases)
  - Camera state buffer
  - Feature map
  - Covariance matrix

## Output

When running in dual mode, the system generates:

1. **academic_comparison.png**: Three-panel plot showing:
   - 3D trajectory comparison (Ground Truth, MSCKF Baseline, SKNet)
   - 2D top-view trajectory
   - Absolute Position Error (APE) over time with RMSE values

2. **trajectory_data.npy**: Raw trajectory data for further analysis

## Testing

A basic integration test is provided in `test_integration.py`:

```bash
python test_integration.py
```

This validates:
- Module imports
- MSCKF initialization with `update_mode` parameter
- SKNetAdapter method signatures
- VIO initialization with `enable_sknet_comparison` flag

## Future Improvements

1. **Configuration File**: Move model path and device settings to config file
2. **Online Comparison**: Real-time trajectory comparison during runtime
3. **Metrics Export**: Export detailed comparison metrics (ATE, RPE, etc.)
4. **Visualization**: Interactive 3D visualization of trajectories
5. **Ablation Studies**: Easy switching between different SKNet architectures

## Troubleshooting

### SKNet Inference Fails
- Check model path and ensure model file exists
- Verify CUDA availability if using GPU (`device='cuda'`)
- Check input dimensions match model expectations

### Trajectories Diverge Significantly
- Verify ground truth alignment (zero-start alignment is applied)
- Check feature tracking quality
- Review SKNet model training data compatibility

### Memory Issues
- Reduce camera state buffer size in config
- Use `device='cpu'` for SKNet if GPU memory is limited
- Process shorter sequences for testing

## References

- MSCKF Paper: [Link to paper]
- SKNet Paper: [Link to paper]
- EuRoC Dataset: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
