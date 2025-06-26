"""
Quick sampling script to test different prompts with trained models
"""
import os
import pickle
import sys
from contextlib import nullcontext
import torch

# Add the models directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from model import GPTConfig, GPT

def sample_from_model(out_dir, start_prompt, num_samples=3, max_new_tokens=200, temperature=0.8):
    """Sample text from a trained model"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'float16' if torch.cuda.is_available() else 'float32'
    
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Load model
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        return
        
    print(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    # Fix state dict keys if needed
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Load meta for encoding/decoding
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('..', 'data', checkpoint['config']['dataset'], 'meta.pkl')
        if os.path.exists(meta_path):
            print(f"Loading meta from {meta_path}")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        else:
            print("No meta.pkl found, cannot encode/decode properly")
            return
    else:
        print("No dataset config found in checkpoint")
        return
    
    # Encode prompt
    start_ids = encode(start_prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    print(f"\\nGenerating {num_samples} samples with prompt: '{start_prompt}'")
    print("=" * 60)
    
    # Generate samples
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=200)
                result = decode(y[0].tolist())
                print(f"\\nSample {k+1}:")
                print("-" * 40)
                print(result)
                print("-" * 40)

if __name__ == "__main__":
    # Test different prompts with the test model
    prompts = [
        "Blood and",
        "Dark shadows",
        "In the depths of",
        "Fire burns",
        "Metal warriors"
    ]
    
    print("Testing various metal-themed prompts...")
    
    for prompt in prompts:
        print(f"\\n{'='*60}")
        sample_from_model('out-darklyrics-test', prompt, num_samples=2, max_new_tokens=150, temperature=0.9)
        print(f"{'='*60}")
