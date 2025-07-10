import unittest
import sys
import os

# Add project root to path for 'models/' imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from setup_tests import BaseModelTest, run_system_info
from stable_video_diffusion_tests import TestStableVideoDiffusion
from animatediff_tests import TestAnimateDiff
from tooncrafter_tests import TestToonCrafter

class TestAllModels(BaseModelTest):
    """Test suite that runs all model tests"""
    pass

if __name__ == '__main__':
    run_system_info()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(BaseModelTest))
    #suite.addTests(loader.loadTestsFromTestCase(TestStableVideoDiffusion))
    #suite.addTests(loader.loadTestsFromTestCase(TestAnimateDiff))
    suite.addTests(loader.loadTestsFromTestCase(TestToonCrafter))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)