"""
Configuration management utilities
"""

import os
import json
from types import SimpleNamespace

def load_config(config_path):
    """Load configuration from a Python file or JSON file"""
    if config_path.endswith('.py'):
        # Load Python config file
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Extract configuration variables
        config_dict = {
            key: value for key, value in vars(config_module).items()
            if not key.startswith('__')
        }
        return SimpleNamespace(**config_dict)
    
    elif config_path.endswith('.json'):
        # Load JSON config file
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return SimpleNamespace(**config_dict)
    
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")

def save_config(config, config_path):
    """Save configuration to a JSON file"""
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def get_default_config():
    """Get default training configuration"""
    return SimpleNamespace(
        # Model parameters
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False,
        
        # Training parameters
        learning_rate=6e-4,
        max_iters=600000,
        weight_decay=1e-1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        
        # Data parameters
        dataset='openwebtext',
        gradient_accumulation_steps=5 * 8,
        batch_size=12,
        block_size=1024,
        
        # System parameters
        device='auto',
        dtype='bfloat16',
        compile=True,
        
        # Evaluation parameters
        eval_interval=2000,
        log_interval=1,
        eval_iters=200,
        eval_only=False,
        always_save_checkpoint=True,
        init_from='scratch',
        
        # Logging
        wandb_log=False,
        wandb_project='nanogpt',
        wandb_run_name='gpt2',
    )
