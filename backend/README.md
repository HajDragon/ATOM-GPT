# NanoGPT Backend

This directory contains all the backend components for the nanoGPT project, organized for clean separation from the planned frontend.

## Directory Structure

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

## Key Components

### Scrapers
- **DarkLyrics Scraper**: Comprehensive metal lyrics dataset collection
- **Dataset Builders**: Tools for creating training datasets
- **Test Scripts**: Validation and debugging tools

### Training Pipeline
- **train.py**: Main training script with distributed training support
- **sample.py**: Text generation and sampling utilities
- **bench.py**: Performance benchmarking and evaluation

### Models
- **model.py**: GPT architecture implementation with various configurations

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
