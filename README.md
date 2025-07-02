# ATOM-GPT

![ATOM-GPT](assets/nanogpt.jpg)

**A**dvanced **T**ransformer **O**ptimized for **M**usical **GPT** - A specialized repository for training and finetuning GPT models on metal lyrics. Built for creative text generation with a focus on metal music content, this implementation provides optimized configurations for various GPU setups and comprehensive tools for metal lyrics-based AI training.

This project includes both a **command-line training interface** and a **modern web application** with ChatGPT-style UI for interactive metal lyrics generation.

![repro124m](assets/gpt2_124M_loss.png)

## Project Structure

This project is organized with a clean separation between backend and frontend components:

```
├── backend/          # All backend components (training, models, data)
│   ├── api/          # Flask API server for web application
│   ├── config/       # Training configurations
│   ├── data/         # Datasets and preprocessing
│   ├── models/       # Model definitions
│   ├── scrapers/     # Data collection tools
│   ├── training/     # Training and sampling scripts
│   ├── utils/        # Utility functions
|   |__ visualization/# Project documentation and visualizations
│   └── notebooks/    # Analysis notebooks
├── frontend/         # React web application
│   ├── public/       # Static assets
│   ├── src/          # React components and logic
│   │   ├── components/   # UI components (chat, completion interfaces)
│   │   ├── utils/        # Frontend utilities
│   │   └── App.tsx       # Main application
│   └── package.json  # Frontend dependencies
├── assets/           # Images and static files
├── LICENSE
└── README.md
```

See [backend/README.md](backend/README.md) for detailed backend documentation.

## Architecture & Visualization

For a comprehensive understanding of the ATOM-GPT architecture and workflow:

- **📊 [ATOM GPT Flowchart](backend/visualization/ATOM%20GPT%20FLOWCHART%20.pdf)** - Hand drawn representation of the training pipeline and model architecture
- **🔄 [Hybrid Flowchart](backend/visualization/Hybrid_flowchart.html)** - Interactive flowchart showing the complete data processing and training workflow

Because the code is optimized for metal lyrics generation, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints for different metal genres and styles.

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm flask flask-cors requests
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3
-  `flask` & `flask-cors` for web API server <3
-  `requests` for HTTP requests <3

For the React frontend:
```bash
cd frontend
npm install
```

## Running the Web Application

ATOM-GPT includes a modern web application with ChatGPT-style interface for interactive metal lyrics generation. The app features both conversational chat and text completion modes.

### Prerequisites

1. **Train or download a model** (see training sections below)
2. **Install dependencies** (see install section above)
3. **Node.js** (for React frontend)

### Quick Start - Web App

**Step 1: Start the Backend API Server**
```bash
cd backend/api
python server.py
```

The backend will:
- Load your trained model (from `backend/training/out-darklyrics/`)
- Start Flask API server on `http://localhost:8000`
- Check for LM Studio connection (optional enhancement)
- Display model information and GPU status

**Step 2: Start the Frontend React App**
```bash
cd frontend
npm start
```

The frontend will:
- Start React development server on `http://localhost:3000`
- Connect to backend API automatically
- Open in your default browser

**Step 3: Use the Application**

Navigate to:
- **Chat Interface**: `http://localhost:3000/` - Conversational AI with chat history
- **Completion Interface**: `http://localhost:3000/completion` - Text completion with metal examples

### Web Application Features

🎸 **Chat Interface**
- ChatGPT-style conversation interface
- Persistent chat history with localStorage
- Sidebar for managing conversations
- Real-time typing effects and progress indicators
- Copy-to-clipboard functionality
- Keyboard shortcuts (Ctrl+Enter to send)

🔥 **Completion Interface** 
- Text completion with metal lyric examples
- Pre-filled metal-themed prompts
- Interactive example buttons
- Progress bars and typing animations
- Adjustable generation settings

⚙️ **Settings & Controls**
- Temperature, max tokens, top-p, repetition penalty
- LM Studio enhancement toggle
- Real-time model and system information
- GPU status and parameter count display

🎨 **Modern UI**
- Dark theme optimized for metal aesthetics
- Responsive design for all screen sizes
- Smooth animations and transitions
- Custom scrollbars and visual effects

### Optional: LM Studio Enhancement

For higher quality outputs, you can optionally connect LM Studio:

1. **Install LM Studio** from [lmstudio.ai](https://lmstudio.ai)
2. **Load a model** in LM Studio (e.g., CodeLlama, Mistral, etc.)
3. **Start the server** in LM Studio on default port (1234)
4. **Enable enhancement** in the web app settings

When LM Studio is connected, the app will show "✨ Enhanced" badges and produce higher quality responses.

### Troubleshooting Web App

**Backend Issues:**
- Ensure model is trained and checkpoint exists in `backend/training/out-darklyrics/`
- Check Python dependencies are installed
- Verify no port conflicts on 8000

**Frontend Issues:**
- Run `npm install` in frontend directory
- Check Node.js version (16+ recommended)
- Verify no port conflicts on 3000

**Connection Issues:**
- Check backend is running on port 8000
- Ensure CORS is properly configured
- Check browser console for error messages

## quick start

### Option 1: Web Application (Recommended)

The easiest way to get started is with the web application. See the "Running the Web Application" section above for detailed instructions.

### Option 2: Command-Line Training & Sampling

### Training on Metal Lyrics Dataset

For training on the comprehensive metal lyrics dataset, first prepare the data:

```sh
python backend/data/DarkLyrics/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT:

```sh
cd backend/training
python train.py ../config/train_darklyrics.py
```

### Training on Metal Lyrics Dataset (Full Power)

For a more substantial training experience using the comprehensive metal lyrics dataset:

**Step 1: Prepare the metal lyrics dataset**
```sh
python backend/data/DarkLyrics/prepare.py
```

This processes your metal lyrics dataset and creates the necessary training files (`train.bin`, `val.bin`, `meta.pkl`).

**Step 2: Train the model**
```sh
cd backend/training
python train.py ../config/train_darklyrics.py
```

This config is optimized for RTX 3050 GPUs (8GB VRAM) and will train a GPT model on metal lyrics. Training will take several hours but will create a model capable of generating lyrics in various metal styles.

**Step 3: Generate lyrics**
```sh
cd backend/training
python sample.py --out_dir=../out-darklyrics
```

This will generate new metal lyrics based on the patterns learned from your dataset.

### Quick Test Run

To verify everything works before starting full training:

```sh
cd backend/training
python train.py ../config/test_darklyrics.py
```

This will train a small model in ~10 minutes and verify your setup is working.

### Hardware Requirements

- **RTX 3050 (8GB)**: Use `train_darklyrics.py` config (optimized)
- **Higher-end GPUs**: You can increase `batch_size`, `n_layer`, `n_embd` in the config
- **CPU only**: Reduce model size significantly and expect much slower training

If you peek inside the config files, you'll see training parameters optimized for different scenarios. The `train_darklyrics.py` config trains a GPT with 8 layers, 8 attention heads, and 512 embedding dimensions - perfect for RTX 3050. Training takes several hours and model checkpoints are saved to `backend/out-darklyrics`. 

**Monitor your training:**
- Watch the terminal for loss values (lower is better)
- Training will automatically save checkpoints
- Use Ctrl+C to stop training early if needed

**Expected Results:**
- Initial loss: ~4.0 (random)
- Good training loss: ~1.5-2.0 
- Training time: 4-8 hours on RTX 3050

## Advanced Usage

### Custom Dataset Training

To train on your own text dataset:

1. **Prepare your data**: Create a `.txt` file with your text
2. **Create data folder**: Put it in `backend/data/your_dataset/input.txt`
3. **Copy prepare script**: Use `backend/data/DarkLyrics/prepare.py` as template
4. **Create config**: Copy and modify `train_darklyrics.py` with your dataset name
5. **Train**: Run training with your new config

### GPU Memory Optimization

If you run into CUDA out-of-memory errors:

1. **Reduce batch_size**: Try 4 or 2 instead of 8
2. **Reduce block_size**: Try 256 or 384 instead of 512  
3. **Reduce model size**: Lower `n_layer`, `n_head`, or `n_embd`
4. **Increase gradient_accumulation_steps**: Compensates for smaller batch_size

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py backend/config/test_darklyrics.py --device=cpu --compile=False
```

Here, since we are running on CPU instead of GPU we must set `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. This will train a small model on the metal lyrics dataset. You can then sample from it:

```sh
python sample.py --out_dir=out-darklyrics-test --device=cpu
```

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

For finetuning on the metal lyrics dataset, you can create a new config that initializes from a pretrained GPT-2 checkpoint. Unlike training from scratch, finetuning can take very little time, e.g. on a single GPU just a few minutes.

To create a finetuning config:
1. Copy `backend/config/train_darklyrics.py` 
2. Add `init_from = 'gpt2'` to initialize from pretrained GPT-2
3. Lower the learning rate (e.g., `learning_rate = 1e-5`)
4. Reduce max_iters (e.g., `max_iters = 1000`)

The finetuned model will generate metal lyrics with better language structure since it starts from a pretrained checkpoint.

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to explore transformer architectures and creative text generation. ATOM-GPT focuses specifically on metal music content generation and provides specialized tools for training on metal lyrics datasets.

📋 **Documentation & Visualizations:**
- Review the [ATOM GPT Flowchart](backend/visualization/ATOM%20GPT%20FLOWCHART%20.pdf) for initial overview
- Explore the [Interactive Hybrid Flowchart](backend/visualization/Hybrid_flowchart.html) for workflow visualization

For questions or discussions about ATOM-GPT, feel free to open issues on the GitHub repository.

## acknowledgements

ATOM-GPT is built upon the foundation of transformer architectures and benefits from the open-source community's contributions to deep learning and natural language processing.

## System Requirements

### Hardware
- **GPU**: RTX 3050 8GB or better (recommended)
- **RAM**: 8GB+ system memory  
- **Storage**: 5GB+ free space
- **CPU**: Multi-core processor for React development

### Software
- **Python**: 3.8+ with PyTorch
- **Node.js**: 16+ for React frontend
- **Operating System**: Windows, macOS, or Linux

### Performance Notes
- **RTX 3050**: Optimal performance, 4-8 hour training
- **Higher-end GPUs**: Faster training, can increase model size
- **CPU only**: Much slower but functional with reduced model size
- **Apple Silicon**: Use `--device=mps` for GPU acceleration

## Dataset Information

### Metal Lyrics Dataset Statistics
- **Total characters**: 20.4 million
- **Vocabulary size**: 905 unique characters (including special symbols)
- **Training tokens**: 18.4 million
- **Validation tokens**: 2.0 million
- **Languages**: Primarily English with some international content

The dataset includes comprehensive metal lyrics from various sources, covering multiple metal genres and spanning different eras of metal music history.

### Training Configurations Available

1. **`test_darklyrics.py`** - Quick test (5-10 minutes on RTX 3050)
   - 4 layers, 256 embedding, 1000 iterations
   - Perfect for testing setup

2. **`train_darklyrics.py`** - Full training (4-8 hours on RTX 3050)  
   - 8 layers, 512 embedding, 10000 iterations
   - Production-quality model

### Output Quality & Enhancement

The web application includes intelligent quality control:

**Base Model Output**: Direct generation from your trained ATOM-GPT model
**Enhanced Output**: Optional LM Studio integration for higher quality
**Smart Fallbacks**: Curated metal-themed completions when model output is low quality

The system automatically detects poor quality generations and provides metal-themed fallback completions to ensure users always receive coherent, thematic responses.

**Quality Indicators:**
- ✨ **Enhanced by LM Studio** - High-quality output from connected LM Studio model
- **Standard Output** - Direct generation from your trained model
- **Fallback Completion** - Curated thematic response when needed
