import unittest
import torch
import sys
import os

# Add project root to path for 'models/' imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from setup_tests import BaseModelTest
from models.stable_video_diffusion.runner import StableVideoDiffusionRunner

class TestStableVideoDiffusion(BaseModelTest):
    
    def test_svd_runner_initialization(self):
        """Test Stable Video Diffusion runner initialization"""
        try:
            svd_runner = StableVideoDiffusionRunner(self.config)
            print("✓ Stable Video Diffusion runner initialized")
            self.assertTrue(True)
        except Exception as e:
            print(f"⚠ SVD runner initialization failed: {e}")
            self.fail(f"SVD runner initialization failed: {e}")

if __name__ == '__main__':
    print("Running Stable Video Diffusion tests...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print("="*50)
    
    unittest.main(verbosity=2)