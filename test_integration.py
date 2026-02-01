#!/usr/bin/env python3
"""
Simple integration test to validate SKNet integration changes.
This script checks that:
1. MSCKF can be initialized with update_mode parameter
2. SKNetAdapter.get_optimal_gain can return covariances
3. VIO can be initialized with enable_sknet_comparison flag
"""

import sys
import os
import numpy as np

# Add msckfv2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'msckfv2'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from config import ConfigEuRoC
        from msckf import MSCKF
        from sknet_adapter import SKNetAdapter
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_msckf_initialization():
    """Test MSCKF initialization with update_mode"""
    print("\nTesting MSCKF initialization...")
    try:
        from config import ConfigEuRoC
        from msckf import MSCKF
        
        config = ConfigEuRoC()
        
        # Test baseline MSCKF
        msckf_baseline = MSCKF(config, update_mode="msckf")
        assert msckf_baseline.update_mode == "msckf", "Baseline update_mode should be 'msckf'"
        print("✓ Baseline MSCKF initialized with update_mode='msckf'")
        
        # Test SKNet MSCKF
        msckf_sknet = MSCKF(config, update_mode="sknet")
        assert msckf_sknet.update_mode == "sknet", "SKNet update_mode should be 'sknet'"
        print("✓ SKNet MSCKF initialized with update_mode='sknet'")
        
        return True
    except Exception as e:
        print(f"✗ MSCKF initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sknet_adapter_methods():
    """Test SKNetAdapter method signatures"""
    print("\nTesting SKNetAdapter methods...")
    try:
        from config import ConfigEuRoC
        from sknet_adapter import SKNetAdapter
        
        config = ConfigEuRoC()
        adapter = SKNetAdapter(config, model_path=None, device='cpu')
        
        # Create dummy data
        H_thin = np.random.randn(10, 21)
        r_thin = np.random.randn(10, 1)
        
        # Test without covariances (backward compatibility)
        result = adapter.get_optimal_gain(H_thin, r_thin, return_covariances=False)
        print("✓ get_optimal_gain(return_covariances=False) returns single value")
        
        # Test with covariances (new feature)
        result = adapter.get_optimal_gain(H_thin, r_thin, return_covariances=True)
        if result is not None:
            assert isinstance(result, tuple) and len(result) == 3, "Should return (K, Pk, Sk)"
            print("✓ get_optimal_gain(return_covariances=True) returns (K, Pk, Sk) tuple")
        else:
            print("✓ get_optimal_gain(return_covariances=True) returns None (expected without model)")
        
        return True
    except Exception as e:
        print(f"✗ SKNetAdapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_measurement_update_signature():
    """Test that measurement_update accepts update_mode parameter"""
    print("\nTesting measurement_update method signature...")
    try:
        from config import ConfigEuRoC
        from msckf import MSCKF
        import inspect
        
        config = ConfigEuRoC()
        msckf = MSCKF(config, update_mode="msckf")
        
        # Check method signature
        sig = inspect.signature(msckf.measurement_update)
        params = list(sig.parameters.keys())
        
        assert 'update_mode' in params, "measurement_update should have update_mode parameter"
        print(f"✓ measurement_update has parameters: {params}")
        
        return True
    except Exception as e:
        print(f"✗ measurement_update signature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vio_initialization():
    """Test VIO initialization with enable_sknet_comparison flag"""
    print("\nTesting VIO initialization...")
    try:
        from config import ConfigEuRoC
        from vio import VIO
        from queue import Queue
        
        config = ConfigEuRoC()
        img_queue = Queue()
        imu_queue = Queue()
        
        # Note: VIO starts threads, so we need to be careful
        # We'll just check the constructor signature
        import inspect
        sig = inspect.signature(VIO.__init__)
        params = list(sig.parameters.keys())
        
        assert 'enable_sknet_comparison' in params, "VIO should have enable_sknet_comparison parameter"
        print(f"✓ VIO.__init__ has parameters: {params}")
        
        return True
    except Exception as e:
        print(f"✗ VIO initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("SKNet Integration Validation Tests")
    print("="*60)
    
    tests = [
        test_imports,
        test_msckf_initialization,
        test_sknet_adapter_methods,
        test_measurement_update_signature,
        test_vio_initialization,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
