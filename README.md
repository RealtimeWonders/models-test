# Image-to-Video Models for Manga Animation (M3 Max Optimized)

This project tests locally runnable image-to-video models optimized for **MacBook Pro M3 Max (48GB RAM)** to find the best approach for generating animated videos from manga panels.

## ✅ Locally Runnable Models

### Memory-Efficient Models (< 10GB VRAM)
- **CogVideoX-2B** (THUDM): 6-8GB, 720P, 16 frames
- **Stable Video Diffusion** (StabilityAI): Quantized versions
- **AnimateDiff** (Runway): Motion adapter approach
- **ToonCrafter** (Doubiiu): 4-6GB, 512x320, 16 frames, **manga-specialized**

### M3 Max Optimizations
- **MPS (Metal Performance Shaders)** acceleration
- **Memory-efficient attention slicing**
- **CPU offloading** for large models  
- **Float32 precision** for MPS compatibility

## 🛠️ Project Structure

```
models/
├── src/
│   ├── local_model_tester.py    # M3 Max optimized testing
│   └── evaluation_metrics.py    # Video quality assessment
├── configs/
│   └── memory_config.py         # Memory optimization settings
├── data/
│   ├── input/                   # Manga images
│   └── output/                  # Generated videos
├── results/                     # Evaluation results
├── quick_test.py               # Compatibility test
└── requirements.txt            # Apple Silicon dependencies
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run compatibility test:**
   ```bash
   python quick_test.py
   ```

3. **Add manga images** to `data/input/`

4. **Run model tests:**
   ```bash
   python src/local_model_tester.py
   ```

5. **Evaluate results:**
   ```bash
   python src/evaluation_metrics.py
   ```

## 📊 Expected Performance

| Model | Memory Usage | Inference Time | Quality | Best For |
|-------|-------------|----------------|---------|----------|
| **ToonCrafter** | 4-6GB | 20-40s | **High** | **Manga/Cartoon** |
| CogVideoX-2B | 6-8GB | 30-60s | High | General I2V |
| Stable Video Diffusion | 4-6GB | 20-40s | Medium | Fast Generation |
| AnimateDiff | 3-5GB | 15-30s | Medium | Text-to-Animation |

## 🔧 Memory Optimizations

- **Attention slicing** for reduced memory usage
- **Sequential CPU offload** for large models
- **Batch size 1** for conservative memory usage
- **Float32 precision** for MPS compatibility
- **Garbage collection** between model tests

## 🎨 Model Highlights

### ToonCrafter - **Perfect for Manga!**
- **Specialized for cartoon/animation** content
- **Interpolation-based**: Creates smooth transitions between frames
- **512x320 resolution** optimized for mobile/web
- **16 frames** (~2 seconds) output
- **Text-guided motion** control
- **Manual installation required** from [GitHub](https://github.com/ToonCrafter/ToonCrafter)

### Key Features:
- Two-image interpolation (start + end frame)
- Text prompt for motion description
- Optimized for cartoon/manga style
- Apache-2.0 license (research friendly)

## 📝 Notes

- Optimized for **48GB RAM** MacBook Pro M3 Max
- Uses **MPS backend** for GPU acceleration
- Models download automatically on first run
- Results saved with memory usage statistics
- **ToonCrafter requires manual setup** from GitHub repository