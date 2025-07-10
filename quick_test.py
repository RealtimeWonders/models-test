#!/usr/bin/env python3
"""
Quick compatibility test for M3 Max MacBook Pro
Tests basic functionality before running full model tests
"""
import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_pytorch_mps():
    """Test PyTorch MPS availability"""
    print("=== PyTorch MPS Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_available():
        # Test basic tensor operations
        device = torch.device("mps")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        print(f"✅ MPS tensor operations work")
        print(f"Result shape: {z.shape}")
        return True
    else:
        print("❌ MPS not available")
        return False

def test_memory_config():
    """Test memory configuration"""
    print("\n=== Memory Configuration Test ===")
    
    try:
        from configs.memory_config import memory_config, get_memory_usage
        
        memory_config.print_config()
        memory_usage = get_memory_usage()
        print(f"Current memory usage: {memory_usage['ram_used_gb']:.2f}GB ({memory_usage['ram_percent']:.1f}%)")
        print(f"Available memory: {memory_usage['available_gb']:.2f}GB")
        
        return True
    except Exception as e:
        print(f"❌ Memory config test failed: {e}")
        return False

def test_diffusers_import():
    """Test diffusers library import"""
    print("\n=== Diffusers Import Test ===")
    
    try:
        import diffusers
        print(f"✅ Diffusers version: {diffusers.__version__}")
        
        # Test basic pipeline import
        from diffusers import StableVideoDiffusionPipeline
        print("✅ StableVideoDiffusionPipeline import successful")
        
        return True
    except Exception as e:
        print(f"❌ Diffusers import failed: {e}")
        return False

def test_transformers_import():
    """Test transformers library import"""
    print("\n=== Transformers Import Test ===")
    
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        return True
    except Exception as e:
        print(f"❌ Transformers import failed: {e}")
        return False

def test_video_processing():
    """Test video processing libraries"""
    print("\n=== Video Processing Test ===")
    
    try:
        import cv2
        import imageio
        from PIL import Image
        
        print(f"✅ OpenCV version: {cv2.__version__}")
        print(f"✅ ImageIO version: {imageio.__version__}")
        print(f"✅ PIL/Pillow available")
        
        return True
    except Exception as e:
        print(f"❌ Video processing test failed: {e}")
        return False

def create_test_directories():
    """Create test directories if they don't exist"""
    print("\n=== Directory Setup Test ===")
    
    dirs_to_create = [
        "data/input",
        "data/output", 
        "results"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/verified: {dir_path}")
    
    return True

def run_compatibility_test():
    """Run all compatibility tests"""
    print("🚀 Starting M3 Max Compatibility Test")
    print("=" * 50)
    
    tests = [
        ("PyTorch MPS", test_pytorch_mps),
        ("Memory Config", test_memory_config),
        ("Diffusers Import", test_diffusers_import),
        ("Transformers Import", test_transformers_import),
        ("Video Processing", test_video_processing),
        ("Directory Setup", create_test_directories),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("🎯 Compatibility Test Results")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Summary: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your M3 Max is ready for I2V model testing.")
        print("\nNext steps:")
        print("1. Add manga images to data/input/")
        print("2. Run: python src/local_model_tester.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("You may need to install missing dependencies:")
        print("pip install -r requirements.txt")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_compatibility_test()
    sys.exit(0 if success else 1)