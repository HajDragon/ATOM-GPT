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

def generate_sentence_completion(prompt, max_tokens=40, temp=0.8, top_p_val=0.9, rep_penalty=1.2):
    """Generate a sentence completion - focused on continuing the given sentence naturally"""
    with torch.no_grad():
        with ctx:
            # For sentence completion, we want a focused, coherent continuation
            # Clean the prompt to ensure it ends properly for continuation
            clean_prompt = prompt.strip()
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

def generate_response(prompt, max_tokens=max_new_tokens, temp=temperature, top_p_val=top_p, rep_penalty=repetition_penalty, max_retries=3):
    """Generate a response from the model with quality filtering and retry logic"""
    
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
                
                # Enhance with LM Studio if available
                response = lm_enhancer.enhance_response(prompt, response, max_tokens)
                
                # If we got a reasonable response, return it (less strict on first attempt)
                if attempt == 0 and response and len(response.split()) >= 3:
                    return response
                elif is_quality_response(response):
                    return response
                
                # If this attempt failed, try again with different parameters
                if attempt == 0:
                    temp = min(temp * 1.1, 1.0)  # Slightly increase temperature
                    rep_penalty = min(rep_penalty * 1.1, 1.5)  # Increase repetition penalty
                elif attempt == 1:
                    temp = min(temp * 0.9, 0.6)  # Decrease temperature for more focus
                    rep_penalty = min(rep_penalty * 1.2, 1.8)  # Further increase repetition penalty
    
    # If all attempts failed, return a fallback response
    return "The shadows whisper of ancient metal, but the words are lost in the void..."

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
                    else:
                        print("LM Studio commands:")
                        print("  /lmstudio - Show connection status")
                        print("  /lmstudio connect - Try to reconnect")
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
    def __init__(self, base_url="http://localhost:8080", fallback_url="http://192.168.56.1:8080"):
        self.base_urls = [base_url, fallback_url]
        self.available = False
        self.active_url = None
        self.custom_instruction = """You are an expert editor specializing in metal lyrics and dark poetry. Your task is to enhance AI-generated text while preserving its metal/gothic essence.

CRITICAL RULES:
1. Fix grammar and improve flow WITHOUT changing the meaning
2. Preserve ALL dark, metal, and gothic themes
3. Keep the same approximate length (±50%)
4. Maintain the poetic/lyrical style
5. If the response doesn't match the user's request, create a better response that does
6. NEVER add disclaimers, explanations, or meta-commentary
7. Return ONLY the enhanced text - nothing else

EXAMPLES:
- "of light Beasts will burn in an icy mountain of." → "Beasts of light will burn upon an icy mountain peak."
- "darkness and the void consume" → "Darkness and the void consume all."

Focus on clarity and natural flow while keeping the metal aesthetic intact. Respond with ONLY the enhanced text."""
        
        self.check_availability()
    
    def check_availability(self):
        """Check if LM Studio is available on any of the configured URLs"""
        for url in self.base_urls:
            try:
                response = requests.get(f"{url}/v1/models", timeout=2)
                if response.status_code == 200:
                    self.available = True
                    self.active_url = url
                    print(f"🔗 LM Studio connected at {url}")
                    return True
            except requests.exceptions.RequestException:
                continue
        
        self.available = False
        self.active_url = None
        print("⚠️  LM Studio not available - using direct output")
        return False
    
    def enhance_response(self, user_prompt: str, model_response: str, max_tokens: int = 100) -> str:
        """Enhance the model response using LM Studio for clarity with token limit"""
        if not self.available:
            return model_response
        
        try:
            # Create a concise, efficient enhancement prompt
            enhancement_prompt = f"""Fix grammar and improve flow while preserving dark/metal themes:

User: "{user_prompt}"
AI: "{model_response}"

Enhanced:"""

            payload = {
                "model": "qwen3-4b",
                "messages": [
                    {"role": "system", "content": "You are a concise editor for metal lyrics. Fix grammar and improve flow while preserving all dark/metal themes. Respond with ONLY the enhanced text, no explanations."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                "temperature": 0.3,  # Lower temperature for consistent editing
                "max_tokens": max_tokens,  # Match the chat token setting
                "stream": False
            }
            
            response = requests.post(
                f"{self.active_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30  # Increased timeout for qwen3-4b model
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result['choices'][0]['message']
                
                # Handle qwen3-4b model response format
                enhanced_text = message.get('content', '').strip()
                
                # If content is empty but reasoning_content exists, use that instead
                if not enhanced_text and 'reasoning_content' in message:
                    reasoning = message['reasoning_content'].strip()
                    # Extract the actual enhancement from reasoning if possible
                    if reasoning:
                        print("⚠️  Using reasoning content as fallback")
                        enhanced_text = reasoning
                
                # Clean up any formatting artifacts from the LLM
                enhanced_text = enhanced_text.replace('ENHANCED RESPONSE:', '').strip()
                enhanced_text = enhanced_text.strip('"\'')  # Remove quotes if wrapped
                
                # Validate enhancement (ensure it's not too different)
                if enhanced_text and self._is_valid_enhancement(model_response, enhanced_text):
                    print("✨ Response enhanced via LM Studio")
                    return enhanced_text
                else:
                    print("⚠️  Enhancement validation failed, using original")
                    # Debug info for troubleshooting
                    if enhanced_text:
                        print(f"   Original length: {len(model_response)}, Enhanced length: {len(enhanced_text)}")
                        print(f"   Enhanced preview: {enhanced_text[:100]}...")
                    return model_response
            else:
                print(f"⚠️  LM Studio error: {response.status_code}")
                return model_response
                
        except requests.exceptions.Timeout as e:
            print(f"⚠️  LM Studio timeout (model taking too long): {e}")
            print("   Try reducing prompt complexity or check model performance")
            return model_response
        except requests.exceptions.RequestException as e:
            print(f"⚠️  LM Studio connection failed: {e}")
            # Try to reconnect for next time
            print("🔄 Attempting to reconnect...")
            if self.check_availability():
                print("✅ Reconnection successful")
            else:
                print("❌ Reconnection failed")
            return model_response
        except Exception as e:
            print(f"⚠️  LM Studio enhancement error: {e}")
            return model_response
    
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
        print(f"📝 Custom instruction updated")
    
    def get_status(self) -> dict:
        """Get current LM Studio status"""
        return {
            "available": self.available,
            "active_url": self.active_url,
            "instruction_length": len(self.custom_instruction)
        }

# Initialize the enhancer
lm_enhancer = LMStudioEnhancer()

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(os.path.join(out_dir, 'ckpt.pt')):
        print(f"❌ Model checkpoint not found in {out_dir}/")
        print("Make sure you've trained a model first!")
        sys.exit(1)
        
    interactive_chat()
