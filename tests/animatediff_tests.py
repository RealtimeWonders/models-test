import unittest
import torch
import sys
import os

# Add project root to path for 'models/' imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from setup_tests import BaseModelTest
from models.animatediff.runner import AnimateDiffRunner

class TestAnimateDiff(BaseModelTest):
    
    def test_animatediff_runner_initialization(self):
        """Test AnimateDiff runner initialization"""
        try:
            animatediff_runner = AnimateDiffRunner(self.config)
            print("✓ AnimateDiff runner initialized")
            self.assertTrue(True)
        except Exception as e:
            print(f"⚠ AnimateDiff runner initialization failed: {e}")
            self.fail(f"AnimateDiff runner initialization failed: {e}")

if __name__ == '__main__':
    print("Running AnimateDiff tests...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print("="*50)
    
    unittest.main(verbosity=2)