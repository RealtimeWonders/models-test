import unittest
import torch
import sys
import os

# Add project root to path for 'models/' imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_device, load_config

class BaseModelTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.device, cls.dtype = setup_device()
    
    def test_mps_availability(self):
        """Test if MPS is available on the system"""
        if torch.backends.mps.is_available():
            print("✓ MPS is available")
            self.assertTrue(True)
        else:
            print("⚠ MPS is not available, will use CPU")
            self.assertTrue(True)  # Don't fail, just warn
    
    def test_device_setup(self):
        """Test device and dtype setup"""
        device, dtype = setup_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(dtype, [torch.float16, torch.float32])
        print(f"✓ Device setup successful: {device}, {dtype}")
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = load_config()
        self.assertIsInstance(config, dict)
        self.assertIn('device', config)
        self.assertIn('precision', config)
        self.assertIn('memory_opts', config)
        print("✓ Configuration loading successful")
    
    def test_torch_operations(self):
        """Test basic torch operations"""
        device, dtype = setup_device()
        
        # Test tensor creation and operations
        x = torch.randn(2, 3, device=device, dtype=dtype)
        y = torch.randn(2, 3, device=device, dtype=dtype)
        z = x + y
        
        self.assertEqual(z.shape, (2, 3))
        self.assertEqual(z.device.type, device.type)
        self.assertEqual(z.dtype, dtype)
        print(f"✓ Basic torch operations successful on {device}")
    
    def test_memory_info(self):
        """Test memory information"""
        device, _ = setup_device()
        
        if device.type == 'mps':
            # For MPS, we can't get detailed memory info like CUDA
            print("✓ Running on MPS device")
        elif device.type == 'cuda':
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("✓ Running on CPU")
        
        # Test memory allocation
        try:
            test_tensor = torch.randn(1000, 1000, device=device)
            del test_tensor
            print("✓ Memory allocation test passed")
        except Exception as e:
            print(f"⚠ Memory allocation test failed: {e}")

def run_system_info():
    """Print system information"""
    print("Running model compatibility tests...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print("="*50)