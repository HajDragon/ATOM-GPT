#!/usr/bin/env python3
"""
Interactive Chat Interface for ATOM-GPT
Allows real-time conversation with your trained metal lyrics model.
"""

import os
import sys
import pickle
from contextlib import nullcontext
import torch
import tiktoken
sys.path.append('../models')
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-darklyrics' # ignored if init_from is not 'resume'
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 150 # number of new tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('..', 'data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

def generate_response(prompt, max_tokens=max_new_tokens, temp=temperature):
    """Generate a response from the model given a prompt"""
    start_ids = encode(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_tokens, temperature=temp, top_k=top_k)
                response = decode(y[0].tolist())
                # Remove the original prompt from the response
                if response.startswith(prompt):
                    response = response[len(prompt):]
                return response.strip()

def print_banner():
    """Print the welcome banner"""
    print("=" * 70)
    print("🤘 WELCOME TO ATOM-GPT INTERACTIVE CHAT 🤘")
    print("=" * 70)
    print("Your trained metal lyrics AI is ready to chat!")
    print("Type your message and get metal-inspired responses.")
    print("")
    print("Commands:")
    print("  /help     - Show this help message")
    print("  /temp X   - Set temperature (0.1-2.0, default 0.8)")
    print("  /tokens X - Set max tokens (50-500, default 150)")
    print("  /quit     - Exit the chat")
    print("  /clear    - Clear the screen")
    print("")
    print("Tips:")
    print("  • Try metal themes: 'darkness', 'fire', 'steel', 'death'")
    print("  • Ask for lyrics: 'Write a verse about...'")
    print("  • Lower temperature for more focused responses")
    print("  • Higher temperature for more creative responses")
    print("=" * 70)

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def interactive_chat():
    """Main interactive chat loop"""
    clear_screen()
    print_banner()
    
    current_temp = temperature
    current_tokens = max_new_tokens
    
    print(f"🔥 Model loaded: {checkpoint.get('iter_num', 'Unknown')} iterations trained")
    print(f"🎸 Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"🌡️  Temperature: {current_temp}")
    print(f"🎯 Max tokens: {current_tokens}")
    print("\nStart chatting! (type /help for commands)\n")
    
    while True:
        try:
            # Get user input
            user_input = input("🎤 You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/quit' or command == '/exit':
                    print("🤘 Thanks for chatting with ATOM-GPT! Stay metal! 🤘")
                    break
                    
                elif command == '/help':
                    print("\n" + "=" * 50)
                    print("📖 ATOM-GPT Commands:")
                    print("  /help     - Show this help")
                    print("  /temp X   - Set temperature (0.1-2.0)")
                    print("  /tokens X - Set max tokens (50-500)")
                    print("  /clear    - Clear screen")
                    print("  /quit     - Exit chat")
                    print("  /status   - Show current settings")
                    print("=" * 50 + "\n")
                    
                elif command == '/clear':
                    clear_screen()
                    print("🔥 ATOM-GPT Chat - Screen Cleared 🔥\n")
                    
                elif command == '/status':
                    print(f"\n📊 Current Settings:")
                    print(f"   Temperature: {current_temp}")
                    print(f"   Max tokens: {current_tokens}")
                    print(f"   Device: {device}")
                    print(f"   Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters\n")
                    
                elif command.startswith('/temp '):
                    try:
                        new_temp = float(command.split()[1])
                        if 0.1 <= new_temp <= 2.0:
                            current_temp = new_temp
                            print(f"🌡️  Temperature set to {current_temp}")
                        else:
                            print("❌ Temperature must be between 0.1 and 2.0")
                    except (IndexError, ValueError):
                        print("❌ Usage: /temp 0.8")
                        
                elif command.startswith('/tokens '):
                    try:
                        new_tokens = int(command.split()[1])
                        if 50 <= new_tokens <= 500:
                            current_tokens = new_tokens
                            print(f"🎯 Max tokens set to {current_tokens}")
                        else:
                            print("❌ Tokens must be between 50 and 500")
                    except (IndexError, ValueError):
                        print("❌ Usage: /tokens 150")
                        
                else:
                    print("❌ Unknown command. Type /help for available commands.")
                    
                continue
            
            # Generate response
            print("🤖 ATOM-GPT: ", end="", flush=True)
            
            try:
                response = generate_response(user_input, current_tokens, current_temp)
                
                # Clean up response
                response = response.replace('\n\n', '\n').strip()
                
                # Split long responses into multiple lines for better readability
                if len(response) > 80:
                    words = response.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) > 80:
                            lines.append(current_line.strip())
                            current_line = word
                        else:
                            current_line += " " + word if current_line else word
                    if current_line:
                        lines.append(current_line.strip())
                    
                    for i, line in enumerate(lines):
                        if i == 0:
                            print(line)
                        else:
                            print("              " + line)  # Indent continuation lines
                else:
                    print(response)
                    
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                print("Try a different prompt or check your model.")
            
            print()  # Add blank line for readability
            
        except KeyboardInterrupt:
            print("\n🤘 Goodbye! Stay metal! 🤘")
            break
        except EOFError:
            print("\n🤘 Goodbye! Stay metal! 🤘")
            break

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(os.path.join(out_dir, 'ckpt.pt')):
        print(f"❌ Model checkpoint not found in {out_dir}/")
        print("Make sure you've trained a model first!")
        sys.exit(1)
        
    interactive_chat()
