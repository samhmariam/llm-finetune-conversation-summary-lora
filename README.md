# LLM Fine-Tuning for Dialogue Summarization using LoRA

A comprehensive project demonstrating fine-tuning of Large Language Models (LLMs) for dialogue summarization using Parameter Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation).

## 🎯 Overview

This project fine-tunes the [FLAN-T5-base](https://huggingface.co/google/flan-t5-base) model on the [DialogSum dataset](https://huggingface.co/datasets/knkarthick/dialogsum) to improve dialogue summarization capabilities. The project demonstrates both:

1. **Full Fine-Tuning**: Traditional approach updating all model parameters
2. **PEFT with LoRA**: Efficient approach updating only a small subset of parameters (more efficient and practical)

### Key Features

- ✅ Zero-shot baseline evaluation
- ✅ Full fine-tuning implementation
- ✅ Parameter Efficient Fine-Tuning (PEFT) with LoRA
- ✅ Comprehensive ROUGE metric evaluation
- ✅ GPU optimization strategies
- ✅ Multiple LoRA configuration options
- ✅ Training performance analysis

## 📊 Results

### Performance Metrics (ROUGE Scores)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|-------|---------|---------|---------|------------|
| **Original (Zero-shot)** | 0.234 | 0.076 | 0.198 | 0.197 |
| **Fine-tuned (Full)** | 0.421 | 0.177 | 0.347 | 0.346 |
| **PEFT (LoRA)** | 0.407 | 0.182 | 0.337 | 0.338 |

*Note: Scores vary based on training configuration. See optimization guides for best practices.*

### Model Efficiency Comparison

| Metric | Full Fine-Tuning | PEFT (LoRA) |
|--------|------------------|-------------|
| **Trainable Parameters** | ~248M (100%) | ~2.4M (0.97%) |
| **Training Time** | Longer | Shorter |
| **GPU Memory** | High | Lower |
| **Performance** | Slightly Better | Nearly Equivalent |

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM (GPU VRAM for training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/samhmariam/llm-finetune-conversation-summary-lora.git
cd llm-finetune-conversation-summary-lora
```

2. **Install dependencies**

Using `uv` (recommended):
```bash
uv sync
```

Or using `pip`:
```bash
pip install -r pyproject.toml
```

### Quick Start

1. **Open the Jupyter notebook**
```bash
jupyter notebook finetune_genai_model_using_lora.ipynb
```

2. **Run the cells sequentially** to:
   - Load the dataset and model
   - Test zero-shot performance
   - Fine-tune the model
   - Train PEFT model with LoRA
   - Evaluate all models

3. **For optimized training**, see Section 4 in the notebook for GPU-optimized configurations.

## 📁 Project Structure

```
.
├── finetune_genai_model_using_lora.ipynb  # Main notebook
├── README.md                               # This file
├── GPU_OPTIMIZATION_GUIDE.md              # Detailed optimization strategies
├── QUICK_IMPROVEMENTS_SUMMARY.md          # Quick reference for improvements
├── pyproject.toml                         # Project dependencies
├── uv.lock                                # Dependency lock file
├── data/
│   └── dialogue-summary-training-results.csv  # Training results
├── finetuned-flan-t5-base/                # Full fine-tuned model
├── peft-results/                          # PEFT model checkpoints
├── peft-results-improved/                 # Optimized PEFT model
├── results/                               # Full fine-tuning checkpoints
└── results-improved/                      # Optimized full fine-tuning
```

## 🔧 Configuration Options

### LoRA Configurations

The project includes three pre-configured LoRA setups:

1. **Efficient** (Low resource, faster training)
   - Rank: 8
   - Alpha: 16
   - Target modules: `["q", "v"]`

2. **Balanced** (Recommended)
   - Rank: 16
   - Alpha: 32
   - Target modules: `["q", "v", "k", "o"]`

3. **High Performance** (Best quality, slower)
   - Rank: 32
   - Alpha: 64
   - Target modules: `["q", "v", "k", "o"]`

### Training Parameters

Key training arguments that can be tuned:

- `num_train_epochs`: Number of training epochs (3-5 for full, 5-10 for PEFT)
- `per_device_train_batch_size`: Batch size per GPU (1-8 depending on memory)
- `gradient_accumulation_steps`: Steps to accumulate gradients (4-16)
- `learning_rate`: Learning rate (2e-5 for full, 1e-4 for PEFT)
- `max_steps`: Maximum training steps (-1 to use epochs)

## 📈 Optimization Guide

### Quick Wins

1. **Remove `max_steps=1` limitation** - Set to `-1` or remove entirely
2. **Use more data** - Increase from 2% to 25-100% of dataset
3. **Train longer** - Use 3-5 epochs for full fine-tuning, 5-10 for PEFT

### For Best Results

See detailed guides:
- [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) - Comprehensive optimization strategies
- [QUICK_IMPROVEMENTS_SUMMARY.md](QUICK_IMPROVEMENTS_SUMMARY.md) - Quick reference guide

### Expected Improvements

With optimizations:
- **PEFT Model**: ROUGE-L = 0.30-0.45 (89-183% improvement)
- **Fine-tuned Model**: ROUGE-L = 0.35-0.50 (85-165% improvement)

## 💡 Key Concepts

### PEFT (Parameter Efficient Fine-Tuning)

Instead of updating all 248M parameters, PEFT methods like LoRA only update a small subset (~2.4M), providing:
- **Faster training** with less compute
- **Lower memory requirements**
- **Easier deployment** (smaller model files)
- **Nearly equivalent performance** to full fine-tuning

### LoRA (Low-Rank Adaptation)

LoRA freezes pre-trained weights and injects trainable rank decomposition matrices into transformer layers, dramatically reducing trainable parameters while maintaining quality.

## 🧪 Evaluation Metrics

The project uses ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics:

- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: ROUGE-L with summary-level normalization

## 🔍 Usage Examples

### Generating Summaries

```python
# Load fine-tuned model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("./finetuned-flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Generate summary
dialogue = "Your dialogue text here..."
prompt = f"summarize the following conversation: {dialogue}\nsummary: "
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

### Using PEFT Model

```python
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

# Load base model and PEFT adapter
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
peft_model = PeftModel.from_pretrained(base_model, "./peft-results")

# Generate summary (same as above)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional PEFT methods (Prefix Tuning, Adapters, etc.)
- Multi-GPU training support
- Quantization experiments (4-bit, 8-bit)
- Different base models (T5-large, BART, etc.)
- Advanced evaluation metrics

## 📚 References

- [FLAN-T5 Paper](https://arxiv.org/abs/2210.11416)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DialogSum Dataset](https://huggingface.co/datasets/knkarthick/dialogsum)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index)
- [Transformers Library](https://huggingface.co/docs/transformers/index)

## 📝 License

This project is part of the Coursera "Generative AI with Large Language Models" course.

## 👤 Author

**Samuel H. Mariam**
- GitHub: [@samhmariam](https://github.com/samhmariam)

## 🙏 Acknowledgments

- DeepLearning.AI and AWS for the Generative AI with LLM course
- Hugging Face for the transformers and PEFT libraries
- The DialogSum dataset creators