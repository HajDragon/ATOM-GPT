# ATOM-GPT Backend

This directory contains all backend components for ATOM-GPT, including training pipelines, model definitions, data processing, and API servers. This guide will take you from zero to a fully functioning ATOM-GPT backend.

## 🚀 Quick Start (0 to 100)

### Step 1: Prerequisites
```bash
# Install Python 3.8+
# Install Git
# Have 8GB+ free disk space
# GPU recommended (RTX 3050 or better)
```

### Step 2: Install Dependencies
```bash
cd backend
pip install torch numpy transformers datasets tiktoken wandb tqdm flask flask-cors requests python-dotenv
```

### Step 3: Prepare Your First Dataset
```bash
# For metal lyrics (recommended for beginners):
cd data/DarkLyrics
python prepare.py

# This creates train.bin and val.bin files
```

### Step 4: Train Your First Model
```bash
cd ../training
python train.py ../config/test_darklyrics.py  # Quick 10-minute test
# OR
python train.py ../config/train_darklyrics.py # Full training (4-8 hours)
```

### Step 5: Generate Text
```bash
python sample.py --out_dir=out-darklyrics
```

### Step 6: Start the API Server (Optional)
```bash
cd ../api
python server.py
```

🎉 **You now have a fully working ATOM-GPT backend!**

## 📂 Directory Structure

```
backend/
├── config/           # Training and evaluation configurations
│   ├── train_*.py    # Training configuration files
│   ├── eval_*.py     # Evaluation configuration files
│   └── finetune_*.py # Fine-tuning configurations
├── data/             # All datasets and data processing
│   ├── DarkLyrics/   # Metal lyrics dataset
│   └── openwebtext/  # OpenWebText dataset
├── models/           # Model definitions and architectures
│   └── model.py      # Main GPT model implementation
├── scrapers/         # Web scraping tools and utilities
│   ├── darklyrics_scraper.py # Main DarkLyrics scraper
│   ├── build_dataset.py     # Dataset building script
│   └── test_*.py     # Scraper testing and debugging
├── training/         # Training, sampling, and benchmarking
│   ├── train.py      # Main training script
│   ├── sample.py     # Text generation/sampling
│   ├── bench.py      # Performance benchmarking
│   └── interactive_*.py # Interactive demos
├── utils/            # Utility functions and helpers
│   ├── config_utils.py   # Configuration utilities
│   ├── data_utils.py     # Data processing utilities
│   └── configurator.py   # Configuration management
├── notebooks/        # Jupyter notebooks for analysis
│   ├── scaling_laws.ipynb     # Scaling law experiments
│   └── transformer_sizing.ipynb # Model sizing analysis
└── out-darklyrics*/  # Training outputs and checkpoints
```

## 🎯 Detailed Workflows

### Training a New Model from Scratch

1. **Prepare Your Dataset**:
   ```bash
   # Copy your text data to data/your_dataset/input.txt
   # Create prepare script (use DarkLyrics/prepare.py as template)
   python prepare.py
   ```

2. **Create Training Config**:
   ```python
   # Copy config/train_darklyrics.py to config/train_your_dataset.py
   # Modify dataset_name and paths
   out_dir = 'out-your-dataset'
   eval_interval = 250
   eval_iters = 200
   # ... adjust parameters for your needs
   ```

3. **Start Training**:
   ```bash
   cd training
   python train.py ../config/train_your_dataset.py
   ```

4. **Monitor Progress**:
   - Watch terminal for loss values (lower is better)
   - Training saves checkpoints automatically
   - Use Ctrl+C to stop early if needed

### Fine-tuning a Pre-trained Model

1. **Create Fine-tuning Config**:
   ```python
   # In your config file:
   init_from = 'gpt2'  # Start from GPT-2
   learning_rate = 1e-5  # Lower learning rate
   max_iters = 1000  # Fewer iterations needed
   ```

2. **Run Fine-tuning**:
   ```bash
   python train.py ../config/finetune_your_dataset.py
   ```

### Setting Up the API Server

1. **Ensure Model Exists**:
   ```bash
   # Check that your model checkpoint exists:
   ls training/out-darklyrics/ckpt.pt
   ```

2. **Configure Environment** (optional):
   ```bash
   cp ../.env.example ../.env
   # Edit .env file with your settings
   ```

3. **Start Server**:
   ```bash
   cd api
   python server.py
   ```

4. **Test API**:
   ```bash
   curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "The darkness of metal", "max_tokens": 50}'
   ```

## 🔧 Configuration Guide

### Hardware Optimization

**RTX 3050 (8GB)**:
```python
batch_size = 8
block_size = 512
n_layer = 8
n_head = 8
n_embd = 512
```

**RTX 4080+ (16GB+)**:
```python
batch_size = 16
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
```

**CPU Only**:
```python
device = 'cpu'
compile = False
batch_size = 1
block_size = 256
n_layer = 4
```

### Training Parameters

- **learning_rate**: Start with 6e-4, reduce if loss plateaus
- **max_iters**: 5000-10000 for good results
- **eval_interval**: 250 (check validation loss every 250 steps)
- **batch_size**: Larger = faster training, more memory usage
- **block_size**: Context length (max sequence length)

## 🚨 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```python
# Reduce these in your config:
batch_size = 4  # or smaller
block_size = 256  # or smaller
gradient_accumulation_steps = 2  # increase this to compensate
```

**ModuleNotFoundError**:
```bash
# Make sure you're in the right directory:
cd backend/training
# Check Python path:
export PYTHONPATH="${PYTHONPATH}:../"
```

**No GPU Detected**:
```bash
# Install CUDA-enabled PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Training Too Slow**:
```python
# Enable compilation (PyTorch 2.0+):
compile = True
# Reduce precision:
dtype = 'bfloat16'  # if supported by GPU
```

## 📊 Performance Expectations

| Model Size | Training Time (RTX 3050) | Quality | Use Case |
|-----------|---------------------------|---------|----------|
| Small (4 layers) | 30 minutes | Basic | Testing/Experimentation |
| Medium (8 layers) | 4-8 hours | Good | Production Ready |
| Large (12 layers) | 12-24 hours | Excellent | Best Quality |

## 🎛️ Advanced Features

### Distributed Training
```bash
# Multi-GPU training:
torchrun --nproc_per_node=4 train.py config/train_darklyrics.py
```

### Weights & Biases Logging
```python
# In your config:
wandb_log = True
wandb_project = 'atom-gpt'
wandb_run_name = 'my-experiment'
```

### Custom Tokenizers
```python
# For custom datasets, modify prepare.py:
# 1. Add your tokenizer
# 2. Update vocab_size in model config
# 3. Ensure encoding/decoding consistency
```

## 📚 Related Documentation

- **[Main Project README](../README.md)** - Overall project setup
- **[Frontend README](../frontend/README.md)** - Web interface setup
- **[Training Updates](training/README_UPDATES.md)** - Latest training improvements
- **[Config Guide](config/ENHANCED_CONFIG_GUIDE.md)** - Advanced configuration
- **[Test Documentation](../tests/README.md)** - Testing framework

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project follows the same license as the main ATOM-GPT repository.

### Data Management
- Organized datasets for different domains (lyrics, literature, web text)
- Preprocessing and tokenization utilities
- Configuration management for different training scenarios

## Usage

### Training a Model
```bash
cd backend/training
python train.py --config ../config/train_darklyrics.py
```

### Generating Text
```bash
cd backend/training
python sample.py --out_dir ../out-darklyrics
```

### Scraping New Data
```bash
cd backend/scrapers
python darklyrics_scraper.py
```

## Future Frontend Integration

This backend is designed to be API-ready for future frontend development. All core functionality is modularized and can be easily exposed through REST endpoints or other interface patterns.

## Development Notes

- All file paths are configured to work within the backend directory structure
- Configuration files are centralized in the `config/` directory
- Data outputs are contained within the backend to maintain clean separation
- Utilities are modularized for easy reuse across components
