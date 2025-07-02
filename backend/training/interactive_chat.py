#!/usr/bin/env python3
"""
Interactive Chat Interface for ATOM-GPT
Allows real-time conversation with your trained metal lyrics model.
"""

import os
import sys
import pickle
import re
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import tiktoken
sys.path.append('../models')
from model import GPTConfig, GPT
import requests
import json
from typing import Optional

# -----------------------------------------------------------------------------
# Text cleaning and filtering functions

def clean_dataset_artifacts(text):
    """Remove dataset artifacts and metadata from generated text"""
    # Remove lyrics metadata patterns
    text = re.sub(r'\[.*?Lyrics.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?not available.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?Album:.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?Artist:.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?Band:.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?Track.*?\]', '', text, flags=re.IGNORECASE)
    
    # Remove band/album patterns that appear inline
    text = re.sub(r'---\s*Band:.*?\|.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Band:.*?\|.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Album:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Artist:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    
    # Remove attribution patterns
    text = re.sub(r'Thanks to .* for .*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Lyrics:.*?(?=\n|\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'vocals\s*:.*?(?=\n|\.|$)', '', text, flags=re.IGNORECASE)  # More specific
    text = re.sub(r'Bass\s*:.*?(?=\n|\.|$)', '', text, flags=re.IGNORECASE)   # More specific  
    text = re.sub(r'Guitar\s*:.*?(?=\n|\.|$)', '', text, flags=re.IGNORECASE) # More specific
    text = re.sub(r'Drums\s*:.*?(?=\n|\.|$)', '', text, flags=re.IGNORECASE)  # More specific
    
    # Remove numeric patterns that look like song numbers or timestamps
    text = re.sub(r'^\d+\s*[-\.\s]*', '', text)
    text = re.sub(r'\d{3,4}\s*[-\s]*', '', text)  # Remove numbers like 664, 1999, etc.
    
    # Remove common dataset separators and artifacts
    text = re.sub(r'^[\[\]\(\)\-\=\*\#\|]+', '', text)  # Remove leading brackets/separators
    text = re.sub(r'[\[\]\(\)\-\=\*\#\|]+$', '', text)  # Remove trailing brackets/separators
    text = re.sub(r'^---\s*', '', text)  # Remove leading dashes
    
    # Remove standalone brackets or separators
    text = re.sub(r'^\s*[\[\]]\s*', '', text)
    text = re.sub(r'^\s*[|]\s*', '', text)
    
    # Remove incomplete words that start with lowercase after capitals (common artifacts)
    text = re.sub(r'\b[A-Z][a-z]+\s+[a-z]\b', lambda m: m.group(0).split()[0], text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()

def has_repetitive_pattern(text, max_repeat=2):
    """Check if text has excessive repetition - now more aggressive"""
    words = text.lower().split()
    if len(words) < 4:
        return False
    
    # Check for repeated sequences of 2-5 words (more aggressive)
    for seq_len in range(2, 6):
        for i in range(len(words) - seq_len * max_repeat):
            sequence = words[i:i+seq_len]
            count = 1
            
            # Count consecutive repetitions
            for j in range(i + seq_len, len(words) - seq_len + 1, seq_len):
                if words[j:j+seq_len] == sequence:
                    count += 1
                else:
                    break
            
            if count >= max_repeat:
                return True
    
    # Also check for single word repetition (very aggressive)
    for i in range(len(words) - 4):
        word = words[i]
        if len(word) > 2:  # Only check meaningful words
            consecutive_count = 1
            for j in range(i + 1, min(i + 6, len(words))):
                if words[j] == word:
                    consecutive_count += 1
                else:
                    break
            if consecutive_count >= 3:  # 3 or more consecutive same words
                return True
    
    return False

def filter_incomplete_sentences(text):
    """Remove incomplete sentences and ensure proper ending"""
    sentences = re.split(r'[.!?]+', text)
    complete_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if sentence seems complete (has subject and verb indicators)
        words = sentence.split()
        if len(words) >= 3:  # Minimum length for a reasonable sentence
            complete_sentences.append(sentence)
    
    if complete_sentences:
        result = '. '.join(complete_sentences)
        # Ensure it ends with proper punctuation
        if not result.endswith(('.', '!', '?')):
            result += '.'
        return result
    
    return text  # Return original if no complete sentences found

def remove_repetitive_phrases(text):
    """Remove repetitive phrases from text"""
    sentences = re.split(r'[.!?]+', text)
    seen_phrases = set()
    cleaned_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if we've seen this exact sentence before
        if sentence.lower() in seen_phrases:
            continue
            
        # Check for very similar sentences (70% word overlap)
        is_similar = False
        sentence_words = set(sentence.lower().split())
        for seen_phrase in seen_phrases:
            seen_words = set(seen_phrase.split())
            if len(sentence_words) > 0 and len(seen_words) > 0:
                overlap = len(sentence_words & seen_words)
                similarity = overlap / max(len(sentence_words), len(seen_words))
                if similarity > 0.7:
                    is_similar = True
                    break
        
        if not is_similar:
            seen_phrases.add(sentence.lower())
            cleaned_sentences.append(sentence)
    
    if cleaned_sentences:
        result = '. '.join(cleaned_sentences)
        if not result.endswith(('.', '!', '?')):
            result += '.'
        return result
    
    return text

def ensure_complete_words(text):
    """Ensure the response ends with complete words, not partial words"""
    if not text or len(text.strip()) < 3:
        return text
        
    # Split into words and check the last word
    words = text.split()
    if not words:
        return text
        
    # Check if the last word seems incomplete (common patterns)
    last_word = words[-1]
    
    # Remove the last word if it seems incomplete
    incomplete_patterns = [
        lambda w: len(w) == 1 and w.isalpha() and w.lower() not in ['a', 'i'],  # Single letters (except 'a', 'i')
        lambda w: w.endswith('-') and len(w) > 1,  # Words ending with hyphen
        lambda w: len(w) >= 2 and w[-1].isalpha() and w[-2:].islower() and not any(x in w for x in ['.', '!', '?', ',', ';']),  # Potential incomplete words
    ]
    
    # Also check if the text ends abruptly mid-word (no proper ending punctuation or space)
    text_ends_abruptly = (
        not text.endswith((' ', '.', '!', '?', ',', ';', ':', '\n')) and
        len(last_word) >= 3 and
        last_word.isalpha()
    )
    
    # Remove incomplete last word if detected
    if text_ends_abruptly or any(pattern(last_word) for pattern in incomplete_patterns):
        # Remove the last incomplete word
        words = words[:-1]
        if words:
            result = ' '.join(words)
            # Ensure proper ending punctuation
            if result and not result.endswith(('.', '!', '?')):
                # Add a period if it ends with a word
                if result[-1].isalpha():
                    result += '.'
            return result
        else:
            return text  # Keep original if removing last word leaves nothing
    
    return text

def is_quality_response(text):
    """Check if the response meets quality criteria - now more strict about repetition"""
    if not text or len(text.strip()) < 8:
        return False
    
    # Check for excessive repetition (much more aggressive)
    if has_repetitive_pattern(text, max_repeat=2):  # Only allow 1 repetition now
        return False
    
    # Check for major dataset artifacts that survived cleaning
    major_artifacts = [
        r'Thanks to .* for',
        r'for sending these lyrics',
        r'for correcting',
        r'Lyrics:',
        r'Band:.*?\|',
        r'Album:',
        r'Artist:',
        r'---.*Band',
    ]
    
    major_artifact_count = sum(1 for pattern in major_artifacts if re.search(pattern, text, re.IGNORECASE))
    if major_artifact_count > 0:  # No major artifacts allowed
        return False
    
    # Check for reasonable word count (more lenient)
    words = text.split()
    if len(words) < 3 or len(words) > 200:
        return False
    
    # Check for coherent word patterns (more lenient)
    if len([w for w in words if len(w) > 1 and w.isalpha()]) < len(words) * 0.5:
        return False
    
    # Additional check for boring repetitive responses
    unique_words = set(words)
    if len(unique_words) < len(words) * 0.6:  # At least 60% unique words
        return False
    
    return True
    """Check if the response meets quality criteria - now more strict about repetition"""
    if not text or len(text.strip()) < 8:
        return False
    
    # Check for excessive repetition (much more aggressive)
    if has_repetitive_pattern(text, max_repeat=2):  # Only allow 1 repetition now
        return False
    
    # Check for major dataset artifacts that survived cleaning
    major_artifacts = [
        r'Thanks to .* for',
        r'for sending these lyrics',
        r'for correcting',
        r'Lyrics:',
        r'Band:.*?\|',
        r'Album:',
        r'Artist:',
        r'---.*Band',
    ]
    
    major_artifact_count = sum(1 for pattern in major_artifacts if re.search(pattern, text, re.IGNORECASE))
    if major_artifact_count > 0:  # No major artifacts allowed
        return False
    
    # Check for reasonable word count (more lenient)
    words = text.split()
    if len(words) < 3 or len(words) > 200:
        return False
    
    # Check for coherent word patterns (more lenient)
    if len([w for w in words if len(w) > 1 and w.isalpha()]) < len(words) * 0.5:
        return False
    
    # Additional check for boring repetitive responses
    unique_words = set(words)
    if len(unique_words) < len(words) * 0.6:  # At least 60% unique words
        return False
    
    return True

# -----------------------------------------------------------------------------
# Configuration
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-darklyrics' # ignored if init_from is not 'resume'
# Convert to absolute path relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, out_dir)
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 60 # number of new tokens generated in each sample (reduced further for better quality)
temperature = 0.7 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random (reduced for less repetition)
top_k = 25 # retain only the top_k most likely tokens (reduced further for better quality)
top_p = 0.8 # nucleus sampling: keep top tokens with cumulative probability <= top_p (reduced for more focus)
repetition_penalty = 1.35 # penalty for repeating tokens (increased significantly to reduce repetition)
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# Find and execute configurator.py from utils
script_dir = os.path.dirname(os.path.abspath(__file__))
configurator_path = os.path.join(script_dir, '..', 'utils', 'configurator.py')
if os.path.exists(configurator_path):
    exec(open(configurator_path).read()) # overrides from command line or config file
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

def generate_sentence_completion(prompt, max_tokens=40, temp=0.8, top_p_val=0.9, rep_penalty=1.2):
    """Generate a sentence completion - focused on continuing the given sentence naturally"""
    with torch.no_grad():
        with ctx:
            # For sentence completion, we want a focused, coherent continuation
            # Clean the prompt to ensure it ends properly for continuation
            clean_prompt = str(prompt).strip()  # Ensure it's a string
            if not clean_prompt.endswith((' ', ',', '...', '-')):
                # Add a space if the prompt doesn't end with natural continuation markers
                if not clean_prompt.endswith(('.', '!', '?')):
                    clean_prompt += ' '
            
            # Encode the prompt
            start_ids = encode(clean_prompt)
            context = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            generated_tokens = []
            
            for token_idx in range(max_tokens):
                # Get logits from model
                idx_cond = context if context.size(1) <= model.config.block_size else context[:, -model.config.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]  # Get last position logits
                
                # Apply repetition penalty (moderate for natural flow)
                if rep_penalty != 1.0 and len(generated_tokens) > 0:
                    for token_id in set(generated_tokens[-20:]):  # Look at last 20 tokens
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= rep_penalty
                        else:
                            logits[0, token_id] *= rep_penalty
                
                # Apply temperature (slightly higher for creativity)
                logits = logits / temp
                
                # Apply top-k filtering (moderate)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k * 2, logits.size(-1)))  # More options for completion
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Apply nucleus (top-p) sampling
                if top_p_val < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p_val
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Convert to probabilities and sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                next_token_id = next_token[0, 0].item()
                generated_tokens.append(next_token_id)
                
                # Add to context
                context = torch.cat((context, next_token), dim=1)
                
                # Check for stopping conditions
                decoded_token = decode([next_token_id])
                
                # Stop at natural sentence endings
                if decoded_token in ['.', '!', '?']:
                    break
                
                # Stop at paragraph breaks or major separators
                if decoded_token in ['\n\n', '<|endoftext|>']:
                    break
                
                # Prevent immediate repetition loops
                if len(generated_tokens) >= 4:
                    if (generated_tokens[-4] == generated_tokens[-2] and 
                        generated_tokens[-3] == generated_tokens[-1]):
                        break
                
                # For sentence completion, prefer to stop at clause boundaries when we have enough content
                if len(generated_tokens) >= 15 and decoded_token in [',', ';', ':']:
                    # Higher chance to stop at natural breaks for longer completions
                    if torch.rand(1).item() < 0.3:
                        break
            
            # Decode the full response
            full_response = decode(context[0].tolist())
            
            # Clean up and return the complete sentence
            completion = full_response.strip()
            completion = clean_dataset_artifacts(completion)
            completion = ensure_complete_words(completion)
            
            # Enhance with LM Studio if available
            completion = lm_enhancer.enhance_response(prompt, completion, max_tokens)
            
            # Ensure it ends with proper punctuation for sentence completion
            if completion and not completion.endswith(('.', '!', '?', ',', ';', ':')):
                # Find the last complete word and add appropriate punctuation
                words = completion.split()
                if words:
                    # Add a period if it seems like a complete thought
                    completion = ' '.join(words) + '.'
            
            return completion

def generate_response(prompt, max_tokens=max_new_tokens, temp=temperature, top_p_val=top_p, rep_penalty=repetition_penalty, max_retries=3, enhance=True):
    """Generate a response from the model with quality filtering and retry logic
    
    Args:
        prompt: The input prompt
        max_tokens: Maximum tokens to generate
        temp: Temperature for sampling
        top_p_val: Top-p sampling threshold
        rep_penalty: Repetition penalty
        max_retries: Maximum number of retry attempts
        enhance: Whether to use LM Studio enhancement if available
    
    Returns:
        tuple: (response_text, was_enhanced)
    """
    
    for attempt in range(max_retries):
        with torch.no_grad():
            with ctx:
                # Try different prompt conditioning approaches
                if attempt == 0:
                    # First attempt: direct prompting
                    conditioned_prompt = prompt
                elif attempt == 1:
                    # Second attempt: more conversational
                    conditioned_prompt = f"Question: {prompt}\nAnswer:"
                else:
                    # Third attempt: creative prompting
                    conditioned_prompt = f"Create something about: {prompt}"
                
                # Encode the prompt
                start_ids = encode(conditioned_prompt)
                context = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
                generated_tokens = []
                
                for token_idx in range(max_tokens):
                    # Get logits from model
                    idx_cond = context if context.size(1) <= model.config.block_size else context[:, -model.config.block_size:]
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :]  # Get last position logits
                    
                    # Apply repetition penalty (more aggressive)
                    if rep_penalty != 1.0 and len(generated_tokens) > 0:
                        for token_id in set(generated_tokens[-50:]):  # Look at last 50 tokens
                            if logits[0, token_id] > 0:
                                logits[0, token_id] /= rep_penalty
                            else:
                                logits[0, token_id] *= rep_penalty
                    
                    # Apply temperature
                    logits = logits / temp
                    
                    # Apply top-k filtering
                    if top_k is not None and top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('inf')
                    
                    # Apply nucleus (top-p) sampling
                    if top_p_val < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p_val
                        # Keep at least one token
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Scatter sorted tensors to original indexing
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = -float('inf')
                    
                    # Convert to probabilities and sample
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Check for natural stopping points
                    next_token_id = next_token[0, 0].item()
                    generated_tokens.append(next_token_id)
                    
                    # Add to context
                    context = torch.cat((context, next_token), dim=1)
                    
                    # Check for stopping conditions
                    decoded_token = decode([next_token_id])
                    
                    # Stop at natural break points
                    if decoded_token in ['\n\n', '<|endoftext|>']:
                        break
                    
                    # ULTRA-AGGRESSIVE REPETITION DETECTION - check for repetition in real-time
                    if len(generated_tokens) >= 4:
                        # Check for immediate 2-token repetition (A B A B)
                        if (len(generated_tokens) >= 4 and
                            generated_tokens[-4] == generated_tokens[-2] and 
                            generated_tokens[-3] == generated_tokens[-1]):
                            break
                        
                        # Check for single token repetition (A A A)
                        if (len(generated_tokens) >= 3 and
                            generated_tokens[-3] == generated_tokens[-2] == generated_tokens[-1]):
                            break
                    
                    if len(generated_tokens) >= 6:
                        # Check for 3-token loops (A B C A B C)
                        if (generated_tokens[-6] == generated_tokens[-3] and 
                            generated_tokens[-5] == generated_tokens[-2] and
                            generated_tokens[-4] == generated_tokens[-1]):
                            break
                    
                    # Check decoded text for word-level repetition every few tokens
                    if len(generated_tokens) >= 8 and len(generated_tokens) % 4 == 0:
                        recent_text = decode(generated_tokens[-12:])
                        recent_words = recent_text.lower().split()
                        if len(recent_words) >= 4:
                            # Look for any 2-word phrase repeated
                            found_repetition = False
                            for i in range(len(recent_words) - 3):
                                phrase = (recent_words[i], recent_words[i+1])
                                for j in range(i + 2, len(recent_words) - 1):
                                    if (recent_words[j], recent_words[j+1]) == phrase:
                                        found_repetition = True
                                        break
                                if found_repetition:
                                    break  # Exit the main generation loop
                    
                    # SMART STOPPING: Check if we're approaching token limit and stop at word boundaries
                    if token_idx >= max_tokens * 0.8:  # When we're 80% through the token limit
                        # Check if current token ends a complete word
                        current_text = decode(generated_tokens)
                        
                        # Stop at sentence boundaries (higher priority when near limit)
                        if decoded_token in ['.', '!', '?']:
                            break
                        
                        # Stop at word boundaries (space or punctuation)
                        if decoded_token in [' ', ',', ';', ':', '-', '\n'] and len(current_text.strip()) > 10:
                            break
                        
                        # If we're very close to limit, stop at any reasonable boundary
                        if token_idx >= max_tokens * 0.95 and decoded_token in [' ', ',', '.', '!', '?', ';', ':', '-', '\n']:
                            break
                    
                    # Stop at sentence boundaries (with higher probability for longer responses)
                    if len(generated_tokens) > 15 and decoded_token in ['.', '!', '?']:
                        stop_probability = min(0.4, len(generated_tokens) / 80)  # Increase chance as response gets longer
                        if torch.rand(1).item() < stop_probability:
                            break
                
                # Decode the full response
                full_response = decode(context[0].tolist())
                
                # Remove the original prompt from the response
                if full_response.startswith(conditioned_prompt):
                    response = full_response[len(conditioned_prompt):]
                elif full_response.startswith(prompt):
                    response = full_response[len(prompt):]
                else:
                    response = full_response
                
                # Clean up the response
                response = response.strip()
                
                # Remove leading punctuation from conditioning
                if response.startswith(':'):
                    response = response[1:].strip()
                
                # Apply cleaning functions
                response = clean_dataset_artifacts(response)
                response = remove_repetitive_phrases(response)  # Remove repetitive phrases
                response = filter_incomplete_sentences(response)
                response = ensure_complete_words(response)  # Ensure complete words
                
                # Enhance with LM Studio if available and requested
                original_response = response
                if enhance:
                    enhanced_response, was_enhanced = lm_enhancer.enhance_response(prompt, response, max_tokens)
                    response = enhanced_response
                else:
                    was_enhanced = False
                
                # If we got a reasonable response, return it (less strict on first attempt)
                if attempt == 0 and response and len(response.split()) >= 3:
                    return response, was_enhanced
                elif is_quality_response(response):
                    return response, was_enhanced
                
                # If this attempt failed, try again with different parameters
                if attempt == 0:
                    temp = min(temp * 1.1, 1.0)  # Slightly increase temperature
                    rep_penalty = min(rep_penalty * 1.1, 1.5)  # Increase repetition penalty
                elif attempt == 1:
                    temp = min(temp * 0.9, 0.6)  # Decrease temperature for more focus
                    rep_penalty = min(rep_penalty * 1.2, 1.8)  # Further increase repetition penalty
    
    # If all attempts failed, return a fallback response
    return "The shadows whisper of ancient metal, but the words are lost in the void...", False

def print_banner():
    """Print the welcome banner"""
    print("=" * 70)
    print("🤘 WELCOME TO ATOM-GPT INTERACTIVE CHAT 🤘")
    print("=" * 70)
    print("Your trained metal lyrics AI is ready to chat!")
    print("Type your message and get metal-inspired responses.")
    print("")
    # Show LM Studio status in banner
    if lm_enhancer.available:
        print("🔗 LM Studio Enhancement: ✅ ACTIVE - Responses enhanced for clarity")
    else:
        print("🔗 LM Studio Enhancement: ⚠️  OFFLINE - Direct model output")
    print("")
    
    print("Commands:")
    print("  /help     - Show this help message")
    print("  /temp X   - Set temperature (0.1-2.0, default 0.7)")
    print("  /tokens X - Set max tokens (20-300, default 100)")
    print("  /topp X   - Set nucleus sampling (0.1-1.0, default 0.9)")
    print("  /repeat X - Set repetition penalty (1.0-2.0, default 1.1)")
    print("  /test     - Test response quality with sample prompts")
    print("  /complete - Switch to sentence completion mode")
    print("  /chat     - Switch to normal chat mode")
    print("  /enhance  - Show LM Studio enhancement status")
    print("  /lmstudio - LM Studio management commands")
    print("  /quit     - Exit the chat")
    print("  /clear    - Clear the screen")
    print("  /status   - Show current settings")
    print("")
    print("🎯 Tips:")
    print("  • Try metal themes: 'darkness', 'fire', 'steel', 'death'")
    print("  • Ask for lyrics: 'Write a verse about...'")
    print("  • Use /complete for sentence completions: 'In the darkness...'")
    print("  • Lower temperature = more focused, higher = more creative")
    print("  • Lower top-p = more focused, higher = more diverse")
    print("  • Higher repetition penalty = less repetitive")
    if lm_enhancer.available:
        print("  • Responses are automatically enhanced via LM Studio")
    else:
        print("  • Install LM Studio for enhanced response clarity")
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
    current_top_p = top_p
    current_rep_penalty = repetition_penalty
    completion_mode = False  # Track if we're in sentence completion mode
    
    print(f"🔥 Model loaded: {checkpoint.get('iter_num', 'Unknown')} iterations trained")
    print(f"🎸 Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"🌡️  Temperature: {current_temp}")
    print(f"🎯 Max tokens: {current_tokens}")
    print(f"🎪 Top-p: {current_top_p}")
    print(f"🔄 Repetition penalty: {current_rep_penalty}")
    print(f"🎭 Mode: {'Sentence Completion' if completion_mode else 'Normal Chat'}")
    print(f"🔗 LM Studio: {'✅ Connected' if lm_enhancer.available else '⚠️  Offline'}")
    print("\nStart chatting! (type /help for commands)\n")
    
    while True:
        try:
            # Get user input
            prompt_symbol = "✏️ Complete" if completion_mode else "🎤 You"
            user_input = input(f"{prompt_symbol}: ").strip()
            
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
                    print("  /tokens X - Set max tokens (20-300)")
                    print("  /topp X   - Set top-p (0.1-1.0)")
                    print("  /repeat X - Set repetition penalty (1.0-2.0)")
                    print("  /complete - Switch to sentence completion mode")
                    print("  /chat     - Switch to normal chat mode")
                    print("  /enhance  - Show LM Studio enhancement status")
                    print("  /lmstudio - LM Studio management commands")
                    print("  /clear    - Clear screen")
                    print("  /quit     - Exit chat")
                    print("  /status   - Show current settings")
                    print("=" * 50 + "\n")
                    
                elif command == '/clear':
                    clear_screen()
                    print("🔥 ATOM-GPT Chat - Screen Cleared 🔥\n")
                    
                elif command == '/test':
                    print("\n🧪 Testing response quality with sample prompts...")
                    test_prompts = [
                        "darkness",
                        "fire and steel", 
                        "write a verse about death",
                        "the ancient ones",
                        "metal"
                    ]
                    
                    for i, prompt in enumerate(test_prompts):
                        print(f"\n📝 Test {i+1}: '{prompt}'")
                        print("🤖 ATOM-GPT: ", end="", flush=True)
                        try:
                            test_response = generate_response(prompt, current_tokens, current_temp, current_top_p, current_rep_penalty)
                            print(test_response[:100] + "..." if len(test_response) > 100 else test_response)
                        except Exception as e:
                            print(f"Error: {e}")
                    print("\n✅ Quality test complete!\n")
                    print(f"\n📊 Current Settings:")
                    print(f"   Temperature: {current_temp}")
                    print(f"   Max tokens: {current_tokens}")
                    print(f"   Top-p: {current_top_p}")
                    print(f"   Repetition penalty: {current_rep_penalty}")
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
                        print("❌ Usage: /temp 0.7")
                        
                elif command.startswith('/tokens '):
                    try:
                        new_tokens = int(command.split()[1])
                        if 20 <= new_tokens <= 300:
                            current_tokens = new_tokens
                            print(f"🎯 Max tokens set to {current_tokens}")
                        else:
                            print("❌ Tokens must be between 20 and 300")
                    except (IndexError, ValueError):
                        print("❌ Usage: /tokens 100")
                        
                elif command.startswith('/topp '):
                    try:
                        new_top_p = float(command.split()[1])
                        if 0.1 <= new_top_p <= 1.0:
                            current_top_p = new_top_p
                            print(f"🎪 Top-p set to {current_top_p}")
                        else:
                            print("❌ Top-p must be between 0.1 and 1.0")
                    except (IndexError, ValueError):
                        print("❌ Usage: /topp 0.9")
                        
                elif command.startswith('/repeat '):
                    try:
                        new_penalty = float(command.split()[1])
                        if 1.0 <= new_penalty <= 2.0:
                            current_rep_penalty = new_penalty
                            print(f"🔄 Repetition penalty set to {current_rep_penalty}")
                        else:
                            print("❌ Repetition penalty must be between 1.0 and 2.0")
                    except (IndexError, ValueError):
                        print("❌ Usage: /repeat 1.1")
                        
                elif command == '/complete':
                    completion_mode = True
                    print("✏️ Switched to sentence completion mode!")
                    print("   Type a sentence and I'll complete it with metal-themed continuations.")
                    print("   Example: 'In the darkness of the void...'")
                    print("   Use /chat to return to normal chat mode.")
                    
                elif command == '/chat':
                    completion_mode = False
                    print("🎤 Switched to normal chat mode!")
                    print("   Type any message and I'll respond with metal-themed content.")
                    print("   Use /complete to switch to sentence completion mode.")
                    
                elif command == '/status':
                    print(f"\n📊 Current Settings:")
                    print(f"   Mode: {'Sentence Completion' if completion_mode else 'Normal Chat'}")
                    print(f"   Temperature: {current_temp}")
                    print(f"   Max tokens: {current_tokens}")
                    print(f"   Top-p: {current_top_p}")
                    print(f"   Repetition penalty: {current_rep_penalty}")
                    print(f"   Device: {device}")
                    print(f"   Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
                    print(f"   LM Studio: {'✅ Connected' if lm_enhancer.available else '❌ Offline'}")
                    if lm_enhancer.available:
                        print(f"   Enhancement URL: {lm_enhancer.active_url}")
                
                elif user_input.startswith('/lmstudio'):
                    if user_input == '/lmstudio':
                        status = lm_enhancer.get_status()
                        print(f"🔗 LM Studio Status:")
                        print(f"   Available: {'✅ Yes' if status['available'] else '❌ No'}")
                        if status['active_url']:
                            print(f"   URL: {status['active_url']}")
                        print(f"   Custom instruction: {status['instruction_length']} characters")
                    elif user_input.startswith('/lmstudio connect'):
                        print("🔄 Checking LM Studio connection...")
                        if lm_enhancer.check_availability():
                            print("✅ LM Studio connection established!")
                        else:
                            print("❌ Could not connect to LM Studio")
                            print("   Make sure LM Studio is running on:")
                            print("   • http://localhost:8080 or")
                            print("   • http://192.168.56.1:8080")
                    elif user_input.startswith('/lmstudio instruction '):
                        new_instruction = user_input[len('/lmstudio instruction '):]
                        lm_enhancer.set_custom_instruction(new_instruction)
                        print("✅ Custom enhancement instruction updated!")
                    elif user_input.startswith('/lmstudio test'):
                        print("🧪 Testing LM Studio relevance checking...")
                        
                        # Test 1: Relevant response (just grammar fix)
                        print("\n📝 Test 1 - Grammar Fix:")
                        test_response1 = lm_enhancer.enhance_response("darkness", "the darkness consume all", 30)
                        print(f"   User: 'darkness'")
                        print(f"   Original: 'the darkness consume all'")
                        print(f"   Enhanced: '{test_response1}'")
                        print(f"   Relevant: {lm_enhancer._is_response_relevant('darkness', 'the darkness consume all')}")
                        
                        # Test 2: Irrelevant response (should create new response)
                        print("\n📝 Test 2 - Relevance Fix:")
                        test_response2 = lm_enhancer.enhance_response("write about fire", "darkness and void eternal", 40)
                        print(f"   User: 'write about fire'")
                        print(f"   Original: 'darkness and void eternal'")
                        print(f"   Enhanced: '{test_response2}'")
                        print(f"   Relevant: {lm_enhancer._is_response_relevant('write about fire', 'darkness and void eternal')}")
                        
                        if (test_response1 != "the darkness consume all" or 
                            test_response2 != "darkness and void eternal"):
                            print("\n✅ Enhancement working!")
                        else:
                            print("\n⚠️  No enhancement occurred - checking connection...")
                            lm_enhancer.check_availability()
                    else:
                        print("LM Studio commands:")
                        print("  /lmstudio - Show connection status")
                        print("  /lmstudio connect - Try to reconnect")
                        print("  /lmstudio test - Test enhancement with sample text")
                        print("  /lmstudio instruction <text> - Set custom instruction")
                    continue

                elif user_input.startswith('/enhance'):
                    if user_input == '/enhance':
                        print(f"🔗 LM Studio Enhancement: {'✅ Enabled' if lm_enhancer.available else '❌ Disabled'}")
                        if lm_enhancer.available:
                            print("All responses are automatically enhanced for clarity while preserving metal themes.")
                            print(f"Active URL: {lm_enhancer.active_url}")
                        else:
                            print("Install and run LM Studio to enable response enhancement.")
                            print("LM Studio will automatically enhance responses without changing their meaning.")
                    continue
                        
                else:
                    print("❌ Unknown command. Type /help for available commands.")
                    
                continue
            
            # Generate response
            if completion_mode:
                print("✏️ Completion: ", end="", flush=True)
            else:
                print("🤖 ATOM-GPT: ", end="", flush=True)
            
            try:
                if completion_mode:
                    # Use sentence completion mode
                    response = generate_sentence_completion(
                        user_input, 
                        max_tokens=min(current_tokens, 50),  # Completions are typically shorter
                        temp=current_temp, 
                        top_p_val=current_top_p, 
                        rep_penalty=current_rep_penalty
                    )
                else:
                    # Use normal chat mode
                    response = generate_response(user_input, current_tokens, current_temp, current_top_p, current_rep_penalty)
                
                # Final cleanup for display
                response = response.strip()
                
                # Check if response is empty or just the fallback
                if not response or response == "The shadows whisper of ancient metal, but the words are lost in the void...":
                    print("🔥 *The flames flicker with untold stories...*")
                    print("   Try adjusting the temperature or being more specific!")
                else:
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

# -----------------------------------------------------------------------------
# LM Studio Integration for Response Enhancement

class LMStudioEnhancer:
    def __init__(self, base_url=None, fallback_url=None):
        # Use environment variables with fallbacks
        default_base = os.getenv('LM_STUDIO_URL', 'http://localhost:1234')
        default_fallback = os.getenv('LM_STUDIO_FALLBACK_URL', 'http://localhost:8080')
        
        self.base_urls = [
            base_url or default_base,
            fallback_url or default_fallback
        ]
        self.available = False
        self.active_url = None
        self.custom_instruction = """You are an expert editor specializing in metal lyrics and dark poetry. Your task is to enhance AI-generated text while ensuring it properly addresses the user's request.

CRITICAL RULES:
1. Check if the AI response actually answers or relates to the user's prompt
2. If the response is off-topic, create a new response that addresses the user's request while keeping metal/gothic themes
3. If the response is relevant but poorly written, fix grammar and improve flow
4. Preserve ALL dark, metal, and gothic themes and atmosphere
5. Keep responses concise and focused (same approximate length ±50%)
6. Maintain the poetic/lyrical style appropriate for metal content
7. NEVER add disclaimers, explanations, or meta-commentary
8. Return ONLY the enhanced/corrected text - nothing else

EXAMPLES:
User: "Write about fire"
Bad AI: "darkness and the void consume all" 
Enhanced: "Flames consume the earth, burning bright with ancient fury"

User: "Tell me about death"
Bad AI: "of light Beasts will burn in an icy mountain of."
Enhanced: "Death calls from shadowed realms, where souls find eternal rest"

Focus on relevance first, then clarity and flow, while keeping the metal aesthetic intact."""
        
        self.check_availability()
    
    def check_availability(self):
        """Check if LM Studio is available on any of the configured URLs"""
        for url in self.base_urls:
            try:
                # Only check if the models endpoint responds - no actual completion test
                response = requests.get(f"{url}/v1/models", timeout=3)
                if response.status_code == 200:
                    self.available = True
                    self.active_url = url
                    print(f"[LM Studio] Connected at {url}")
                    return True
                else:
                    print(f"[Warning] LM Studio at {url} not responding properly")
                        
            except requests.exceptions.RequestException:
                continue
        
        self.available = False
        self.active_url = None
        print("[Warning] LM Studio not available - using direct output")
        return False
    
    def enhance_response(self, user_prompt: str, model_response: str, max_tokens: int = 100) -> tuple[str, bool]:
        """
        Enhance the model response using LM Studio for relevance and clarity.
        
        Returns:
            tuple: (enhanced_text, was_actually_enhanced)
        
        This function:
        1. Checks if the ATOM-GPT response actually addresses the user's prompt
        2. If irrelevant, creates a new metal-themed response that does address it
        3. If relevant but poorly written, fixes grammar and flow
        4. Preserves all dark/metal themes and atmosphere
        """
        # Do a real-time availability check before attempting to enhance
        if not self.is_really_available():
            return model_response, False
        
        original_response = model_response
        
        # Try different models in order of preference
        models_to_try = ["phi-2", "meta-llama-3.1-8b-instruct", "qwen3-4b"]
        
        for model_name in models_to_try:
            try:
                # Create clean prompts without instruction words
                if self._is_response_relevant(user_prompt, model_response):
                    # Response is relevant, just fix grammar/flow by passing it directly
                    enhancement_prompt = model_response
                    system_message = "Fix grammar and improve flow. Keep the dark metal theme intact. Return only the improved text."
                else:
                    # Response is not relevant, create new response about the topic
                    enhancement_prompt = user_prompt
                    system_message = "Create dark metal themed content. Be gothic and atmospheric. Return only the content."

                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": enhancement_prompt}
                    ],
                    "temperature": 0.5,  # Lower for more focused responses
                    "max_tokens": min(max_tokens // 2, 50),    # Use half of user's setting, max 50 for LM Studio
                    "stream": False,
                    "stop": ["\n", "User:", "System:", "Assistant:"]  # Reduced stop tokens for longer responses
                }
                
                response = requests.post(
                    f"{self.active_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=10  # Shorter timeout for faster response
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Debug: Check the actual response structure
                    if 'choices' in result and len(result['choices']) > 0:
                        choice = result['choices'][0]
                        message = choice.get('message', {})
                        
                        # Handle different response formats
                        enhanced_text = message.get('content', '').strip()
                        
                        # If no content, try other fields
                        if not enhanced_text:
                            enhanced_text = choice.get('text', '').strip()
                        
                        # Clean up the response more aggressively
                        if enhanced_text:
                            # Remove any unwanted prefixes/suffixes
                            enhanced_text = enhanced_text.strip()
                            
                            # Remove numbered lists, bullet points, and formatting
                            import re
                            enhanced_text = re.sub(r'^\d+\.\s*["\']?', '', enhanced_text)  # Remove "1. " or '1. "'
                            enhanced_text = re.sub(r'^[\-\*]\s*["\']?', '', enhanced_text)  # Remove "- " or "* "
                            
                            # Take only the first meaningful line OR combine multiple lines if they're coherent
                            lines = [line.strip() for line in enhanced_text.split('\n') if line.strip()]
                            if lines:
                                # If user wants longer responses (high max_tokens), combine multiple lines
                                if max_tokens > 60 and len(lines) > 1:
                                    # Combine up to 3 coherent lines for longer responses
                                    combined_lines = []
                                    for line in lines[:3]:
                                        if len(' '.join(combined_lines + [line])) <= max_tokens * 6:  # Rough char limit
                                            combined_lines.append(line)
                                        else:
                                            break
                                    enhanced_text = ' '.join(combined_lines) if combined_lines else lines[0]
                                else:
                                    enhanced_text = lines[0]
                            
                            # Remove quotes, brackets, and extra punctuation from start/end
                            enhanced_text = enhanced_text.strip('"\'()[]{}.,!?-*')
                            
                            # Remove any remaining colons or instruction-like formatting
                            if ':' in enhanced_text:
                                parts = enhanced_text.split(':')
                                # Take the part after the colon if it looks like actual content
                                if len(parts) > 1 and len(parts[1].strip()) > 3:
                                    enhanced_text = parts[1].strip()
                                else:
                                    enhanced_text = parts[0].strip()
                            
                            # Final cleanup - ensure clean start
                            enhanced_text = enhanced_text.strip()
                            
                            # Basic validation - more lenient for longer responses
                            if (enhanced_text and 
                                len(enhanced_text) > 3 and 
                                len(enhanced_text) < max(max_tokens * 8, 300) and  # Allow longer based on token setting
                                not any(phrase in enhanced_text.lower() for phrase in ['i cannot', 'as an ai', 'sorry', 'fix grammar', 'write metal', 'dark metal style'])):
                                # Check if the enhancement actually changed something meaningful
                                if enhanced_text.strip() != original_response.strip():
                                    return enhanced_text, True  # Success with this model and actually enhanced
                                else:
                                    return original_response, False  # No meaningful change
                
                # If this model didn't work, try the next one
                continue
                    
            except requests.exceptions.RequestException:
                # Connection issue with this model, try next
                continue
            except Exception:
                # Other error with this model, try next
                continue
        
        # If all models failed, return original with no enhancement
        return original_response, False
    
    def _is_response_relevant(self, user_prompt: str, model_response: str) -> bool:
        """Check if the model response is relevant to the user prompt using keyword matching"""
        # Convert to lowercase for comparison
        prompt_lower = user_prompt.lower()
        response_lower = model_response.lower()
        
        # Extract key topics from user prompt
        topic_keywords = {
            'fire': ['fire', 'flame', 'burn', 'blaze', 'inferno', 'heat'],
            'death': ['death', 'die', 'dead', 'grave', 'tomb', 'corpse', 'soul'],
            'darkness': ['dark', 'shadow', 'black', 'night', 'void'],
            'metal': ['metal', 'steel', 'iron', 'forge', 'blade'],
            'war': ['war', 'battle', 'fight', 'blood', 'sword', 'warrior'],
            'love': ['love', 'heart', 'romantic', 'passion'],
            'nature': ['forest', 'mountain', 'sea', 'sky', 'earth', 'wind'],
            'power': ['power', 'strength', 'might', 'force', 'dominion']
        }
        
        # Check if user is asking for specific content
        for topic, keywords in topic_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                # User is asking about this topic
                if any(keyword in response_lower for keyword in keywords):
                    return True  # Response contains relevant keywords
                else:
                    return False  # Response doesn't match the topic
        
        # If no specific topic detected, check general relevance
        # Look for common question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'write', 'tell', 'describe']
        if any(word in prompt_lower for word in question_words):
            # It's a question/request, so the response should be substantive
            if len(model_response.split()) < 3:
                return False  # Too short to be a proper answer
        
        return True  # Default to relevant if we can't determine otherwise
    
    def _is_valid_enhancement(self, original: str, enhanced: str) -> bool:
        """Validate that the enhancement is reasonable - very lenient for better responses"""
        # Basic sanity checks
        if not enhanced or len(enhanced.strip()) < 5:
            return False
        
        # Very lenient length check - allow up to 300% for short responses that need expansion
        original_length = len(original)
        enhanced_length = len(enhanced)
        
        # For very short responses (under 50 chars), allow much more expansion
        if original_length < 50:
            max_allowed = original_length * 4  # 400% for very short responses
        # Simple length check - allow reasonable expansion
        max_allowed = max(original_length * 2, 100)  # At least 100 chars allowed
        
        if enhanced_length > max_allowed:
            return False
        
        # Quick check for obvious AI artifacts
        enhanced_lower = enhanced.lower()
        ai_artifacts = ["i cannot", "as an ai", "disclaimer", "sorry, but"]
        
        if any(artifact in enhanced_lower for artifact in ai_artifacts):
            return False
        
        # Basic coherence check
        words = enhanced.split()
        if len(words) < 2:
            return False
        
        return True
    
    def set_custom_instruction(self, instruction: str):
        """Allow users to customize the enhancement instruction"""
        self.custom_instruction = instruction
    
    def get_status(self) -> dict:
        """Get current LM Studio status with real-time check"""
        # Re-check availability in real-time
        self.check_availability()
        return {
            "available": self.available,
            "active_url": self.active_url,
            "instruction_length": len(self.custom_instruction)
        }
    
    def is_really_available(self) -> bool:
        """Perform a quick real-time check to see if LM Studio is actually running"""
        if not self.active_url:
            # Only try to reconnect if we haven't checked recently
            return False
            
        try:
            # Quick test with very short timeout - NO completion test
            response = requests.get(f"{self.active_url}/v1/models", timeout=1)
            is_available = response.status_code == 200
            
            # Update internal state based on real check
            if not is_available:
                self.available = False
                self.active_url = None
                    
            return is_available
                
        except requests.exceptions.RequestException:
            # If we can't reach it, update our status
            self.available = False
            self.active_url = None
            return False

# Initialize the enhancer
lm_enhancer = LMStudioEnhancer()

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(os.path.join(out_dir, 'ckpt.pt')):
        print(f"❌ Model checkpoint not found in {out_dir}/")
        print("Make sure you've trained a model first!")
        sys.exit(1)
        
    interactive_chat()

def generate_metal_completion(prompt, max_tokens=max_new_tokens, temp=0.9, enhance=True):
    """Generate metal-themed lyrical completions with improved quality
    
    Args:
        prompt: The input prompt (metal lyric beginning)
        max_tokens: Maximum tokens to generate
        temp: Temperature for sampling (higher for creativity)
        enhance: Whether to use LM Studio enhancement
    
    Returns:
        tuple: (completion_text, was_enhanced)
    """
    
    # Metal-themed conditioning prefixes to improve output quality
    metal_conditioners = [
        "",  # Direct continuation
        "Epic metal verse: ",
        "Dark metal lyrics: ",
        "Gothic poetry: ",
        "Metal anthem: "
    ]
    
    best_completion = ""
    best_quality = 0
    was_enhanced = False
    
    for attempt, conditioner in enumerate(metal_conditioners):
        try:
            with torch.no_grad():
                with ctx:
                    # Condition the prompt for metal lyrics
                    conditioned_prompt = f"{conditioner}{prompt}".strip()
                    
                    # Encode the prompt
                    start_ids = encode(conditioned_prompt)
                    context = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
                    generated_tokens = []
                    
                    for token_idx in range(max_tokens):
                        # Get logits from model
                        idx_cond = context if context.size(1) <= model.config.block_size else context[:, -model.config.block_size:]
                        logits, _ = model(idx_cond)
                        logits = logits[:, -1, :]
                        
                        # Apply stronger repetition penalty for lyrics
                        if len(generated_tokens) > 0:
                            for token_id in set(generated_tokens[-30:]):
                                if logits[0, token_id] > 0:
                                    logits[0, token_id] /= 1.3  # Stronger penalty
                                else:
                                    logits[0, token_id] *= 1.3
                        
                        # Apply temperature
                        logits = logits / temp
                        
                        # Apply top-k filtering (more restrictive for quality)
                        if top_k is not None and top_k > 0:
                            v, _ = torch.topk(logits, min(40, logits.size(-1)))  # More restrictive
                            logits[logits < v[:, [-1]]] = -float('inf')
                        
                        # Apply nucleus sampling
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > 0.85  # More restrictive
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits[indices_to_remove] = -float('inf')
                        
                        # Sample next token
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        next_token_id = next_token[0, 0].item()
                        generated_tokens.append(next_token_id)
                        context = torch.cat((context, next_token), dim=1)
                        
                        # Check for stopping conditions
                        decoded_token = decode([next_token_id])
                        if decoded_token in ['\n\n', '<|endoftext|>']:
                            break
                        
                        # Stop on repetition
                        if len(generated_tokens) >= 4:
                            if (generated_tokens[-4] == generated_tokens[-2] and 
                                generated_tokens[-3] == generated_tokens[-1]):
                                break
                    
                    # Decode the response
                    response = decode(generated_tokens)
                    
                    # Remove conditioning prefix if it appears in output
                    if conditioner and response.startswith(conditioner):
                        response = response[len(conditioner):].strip()
                    
                    # Clean the response
                    response = clean_dataset_artifacts(response)
                    response = remove_repetitive_phrases(response)
                    response = filter_incomplete_sentences(response)
                    
                    # Filter out meta-commentary and non-lyrical content
                    response = filter_meta_commentary(response)
                    
                    # Quality scoring for metal lyrics
                    quality_score = score_metal_lyric_quality(response)
                    
                    if quality_score > best_quality and len(response.split()) >= 8:
                        best_completion = response
                        best_quality = quality_score
                        
                        # If we get a really good score, we can break early
                        if quality_score > 100:
                            break
                
        except Exception as e:
            print(f"Metal completion attempt {attempt} failed: {e}")
            continue
    
    # Enhance with LM Studio if available and requested
    if enhance and best_completion:
        enhanced_response, was_enhanced = lm_enhancer.enhance_response(prompt, best_completion, max_tokens)
        if enhanced_response and len(enhanced_response.strip()) > len(best_completion.strip()):
            best_completion = enhanced_response
    
    return best_completion.strip(), was_enhanced

def filter_meta_commentary(text):
    """Remove meta-commentary and non-lyrical content"""
    
    # Remove common meta-commentary phrases
    meta_phrases = [
        "this is a great start", "let's make it", "this response is not acceptable",
        "we can improve", "here's a better version", "let me try again",
        "that's not quite right", "we need to", "i think", "perhaps we",
        "maybe we should", "this could be better", "let me", "we should",
        "i would", "you could", "it might", "it would be", "it seems",
        "it appears", "it looks like", "i suggest", "my suggestion"
    ]
    
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        line_lower = line_clean.lower()
        
        # Skip lines with meta-commentary
        contains_meta = any(phrase in line_lower for phrase in meta_phrases)
        
        # Skip lines that start with common meta patterns
        meta_starters = ['this is', 'let\'s', 'we can', 'here\'s', 'that\'s', 'i think',
                        'i would', 'you could', 'it might', 'it would', 'it seems',
                        'perhaps', 'maybe', 'probably', 'i suggest']
        starts_with_meta = any(line_lower.startswith(starter) for starter in meta_starters)
        
        # Skip very short fragments or single words
        if len(line_clean.split()) < 3:
            continue
            
        # Skip lines that look like incomplete sentences
        if line_clean.endswith((' t.', ' s.', ' d.', ' n.', ' r.')):
            continue
            
        if not contains_meta and not starts_with_meta:
            filtered_lines.append(line_clean)
    
    result = ' '.join(filtered_lines).strip()
    
    # Additional cleaning for sentence fragments
    sentences = result.split('.')
    clean_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) >= 4:  # Must have at least 4 words
            clean_sentences.append(sentence)
    
    if clean_sentences:
        result = '. '.join(clean_sentences)
        if not result.endswith('.'):
            result += '.'
    
    return result

def score_metal_lyric_quality(text):
    """Score the quality of metal lyrics"""
    if not text or len(text.strip()) < 10:
        return 0
    
    score = 50  # Base score
    text_lower = text.lower()
    words = text_lower.split()
    
    # Positive indicators for metal lyrics
    metal_words = [
        'darkness', 'shadow', 'fire', 'flame', 'blood', 'steel', 'iron', 
        'thunder', 'storm', 'lightning', 'death', 'doom', 'fate', 'hell',
        'demon', 'beast', 'dragon', 'sword', 'blade', 'battle', 'war',
        'night', 'moon', 'star', 'void', 'abyss', 'eternal', 'ancient',
        'soul', 'spirit', 'power', 'rage', 'fury', 'wrath', 'vengeance',
        'crown', 'throne', 'kingdom', 'empire', 'rise', 'fall', 'dawn',
        'chains', 'gates', 'burning', 'crimson', 'frozen', 'cursed',
        'unholy', 'sacred', 'divine', 'mortal', 'immortal', 'legend'
    ]
    
    # Count metal-themed words (higher value)
    word_count = sum(1 for word in metal_words if word in text_lower)
    score += word_count * 15
    
    # Bonus for proper sentence structure
    if text.count('.') >= 1:
        score += 20
    if text.count(',') >= 1:
        score += 10
    
    # Penalty for common low-quality patterns
    quality_killers = [
        'thought and flesh', 'wind creatures', 'started from below',
        'battle cries for what', 'purpose is in', 'break down t',
        'towards their glory'
    ]
    
    for killer in quality_killers:
        if killer in text_lower:
            score -= 100
    
    # Penalty for sentence fragments
    if text.endswith((' t.', ' s.', ' d.', ' n.', ' r.')):
        score -= 50
    
    # Penalty for very short output
    if len(words) < 8:
        score -= 30
    
    # Bonus for longer, more descriptive content
    if len(words) > 15:
        score += 20
    
    # Penalty for repetitive words
    unique_words = len(set(words))
    if len(words) > 5 and unique_words / len(words) < 0.7:
        score -= 40
    
    # Big penalty for meta-commentary residue
    meta_residue = ['this', 'let\'s', 'we', 'should', 'could', 'might', 'would']
    meta_count = sum(1 for word in meta_residue if word in words)
    score -= meta_count * 25
    
    return max(0, score)

def clean_metal_completion(text, original_prompt):
    """Enhanced cleaning specifically for metal lyric completions"""
    
    # Remove the original prompt if it appears at the start
    if text.startswith(original_prompt):
        text = text[len(original_prompt):].strip()
    
    # Remove common artifacts
    artifacts = [
        'spe.', 'bur.', 'ruin.', 'were bur.', 'cave, spe.', 
        'Sea of winds blow through my cave',
        'thought and flesh', 'wind creatures started from below',
        'battle cries for what the purpose is'
    ]
    
    for artifact in artifacts:
        text = text.replace(artifact, '')
    
    # Split into sentences and clean
    sentences = []
    for sentence in text.split('.'):
        sentence = sentence.strip()
        
        # Skip very short fragments
        if len(sentence.split()) < 4:
            continue
            
        # Skip sentences with common low-quality patterns
        skip_patterns = ['thought and', 'wind creatures', 'sea of winds', 'were bur', 'cave, spe']
        if any(pattern in sentence.lower() for pattern in skip_patterns):
            continue
            
        # Clean up incomplete endings
        if sentence.endswith((' t', ' s', ' d', ' n', ' r', ' spe', ' bur')):
            # Try to find the last complete word
            words = sentence.split()
            if len(words) > 1:
                sentence = ' '.join(words[:-1])
        
        if sentence and len(sentence.split()) >= 4:
            sentences.append(sentence)
    
    if sentences:
        result = '. '.join(sentences)
        if not result.endswith('.'):
            result += '.'
        return result.strip()
    
    return text.strip()

# ...existing code...
