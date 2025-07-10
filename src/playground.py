#!/usr/bin/env python3
"""
Playground for testing model runners
"""

import sys
import os
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from utils import load_config
from models.tooncrafter.runner import ToonCrafterRunner

def test_tooncrafter():
    """Test ToonCrafter runner"""
    print("Testing ToonCrafter...")
    
    # Load config
    config = load_config()
    print("✓ Config loaded")
    
    # Initialize runner
    runner = ToonCrafterRunner(config)
    print("✓ Runner initialized")
    
    # Get model info
    info = runner.get_model_info()
    print("\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with sample images
    sample_images = ["image1.jpg", "image2.jpg"]
    frames = runner.generate_video(sample_images)
    print(f"\nGenerated {len(frames)} frames")

if __name__ == "__main__":
    test_tooncrafter()