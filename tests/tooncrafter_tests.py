import unittest
import torch
import sys
import os

# Add project root to path for 'models/' imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from setup_tests import BaseModelTest
from models.tooncrafter.runner import ToonCrafterRunner

class TestToonCrafter(BaseModelTest):
    
    def test_tooncrafter_runner_initialization(self):
        """Test ToonCrafter runner initialization"""
        try:
            tooncrafter_runner = ToonCrafterRunner(self.config)
            print("✓ ToonCrafter runner initialized")
            self.assertTrue(True)
        except Exception as e:
            print(f"⚠ ToonCrafter runner initialization failed: {e}")
            self.fail(f"ToonCrafter runner initialization failed: {e}")

if __name__ == '__main__':
    print("Running ToonCrafter tests...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print("="*50)
    
    unittest.main(verbosity=2)