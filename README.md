# Models Project: Local Runner for Image-to-Video Hugging Face Models on M3 Max MacBook Pro

This project enables local execution of various image-to-video models from Hugging Face on a MacBook Pro with M3 Max chip and 48GB RAM. It leverages Apple's Metal Performance Shaders (MPS) backend in PyTorch for efficient GPU acceleration.

## Features

- **Modularity**: Each model has its own subdirectory with isolated code
- **Extensibility**: Easy to add new models by implementing standard interface
- **M3 Max Optimization**: Uses MPS device with memory-efficient techniques
- **Multiple Models**: Supports Stable Video Diffusion, AnimateDiff, and ToonCrafter
- **Training Support**: Basic fine-tuning scripts per model using LoRA
- **Evaluation**: Built-in video quality metrics (PSNR, SSIM)

## Project Structure

```
models/
├── README.md
├── environment.yml
├── src/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── utils.py             # Shared utilities
│   └── evaluation.py        # Video quality metrics
├── models/
│   ├── stable_video_diffusion/
│   ├── animatediff/
│   └── tooncrafter/
├── configs/
│   └── global_config.yaml   # Global configuration
├── data/
│   ├── input/               # Input images
│   └── output/              # Generated videos
└── tests/
    └── test_models.py       # Compatibility tests
```

## Setup

1. **Install Conda** (if not already installed):
   - Download from [conda.io](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
   - Or install Miniconda for a lightweight version

2. **Create Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate models-env
   ```

3. **Run Compatibility Test**:
   ```bash
   python tests/test_models.py
   ```

4. **Download Models**: Models auto-download on first run via Diffusers/HF. For ToonCrafter, manually download from https://github.com/ToonCrafter/ToonCrafter

### Environment Management

```bash
# Update environment
conda env update -f environment.yml

# Remove environment
conda env remove -n models-env

# Export current environment
conda env export > environment.yml

# List environments
conda env list
```

## Usage

### Generate Video
```bash
# Using Stable Video Diffusion
python src/main.py --model stable_video_diffusion --input data/input/image.png --output data/output/video.mp4

# Using AnimateDiff
python src/main.py --model animatediff --input data/input/image.png --output data/output/video.mp4

# Using ToonCrafter
python src/main.py --model tooncrafter --input data/input/image.png --output data/output/video.mp4
```

### Training
```bash
python src/main.py --model stable_video_diffusion --train --dataset_path data/input/ --epochs 10
```

### Evaluation
```bash
python src/evaluation.py --video data/output/video.mp4 --reference data/input/reference.mp4
```

## Configuration

Edit `configs/global_config.yaml` to adjust:
- Device settings (mps/cpu)
- Memory optimization options
- Default generation parameters
- Input/output paths

## Adding New Models

1. Create `models/new_model/` directory
2. Implement `runner.py` with `generate_video()`, `train()`, and `save_video()` methods
3. Add model config in `config.yaml`
4. Register in `src/main.py` MODEL_REGISTRY
5. Update requirements.txt if needed

## Hardware Requirements

- MacBook Pro with M3 Max chip
- 48GB RAM (recommended)
- macOS with PyTorch MPS support

## Memory Optimization

- Uses float16 precision
- Enables CPU offloading
- VAE slicing and tiling
- Monitors memory usage with `torch.mps.current_allocated_memory()`

## Troubleshooting

- If MPS is not available, the system will fall back to CPU
- For memory issues, reduce batch size or enable more aggressive offloading
- Check compatibility with `python tests/test_models.py`

## License

This project is for educational and research purposes. Please respect the licenses of individual models and dependencies.