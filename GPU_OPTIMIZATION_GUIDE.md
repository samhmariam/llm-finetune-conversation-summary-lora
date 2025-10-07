# GPU Optimization Guide: Improving ROUGE Scores for LLM Fine-tuning

## Current Status Analysis

Based on your notebook execution, here are the current ROUGE scores:

- **Original Model (untrained)**: All metrics = 0.0 (only 1 training step was used)
- **Fine-tuned Model**: 
  - ROUGE-1: 0.229
  - ROUGE-2: 0.056
  - ROUGE-L: 0.189
  - ROUGE-Lsum: 0.188
- **PEFT Model**:
  - ROUGE-1: 0.198
  - ROUGE-2: 0.056
  - ROUGE-L: 0.159
  - ROUGE-Lsum: 0.159

## Key Issues Identified

1. **Severely Limited Training**: `max_steps=1` was set for both models - essentially no real training occurred
2. **Heavy Data Subsampling**: Dataset filtered to every 50th example (only 2% of data used)
3. **Small Batch Size**: `per_device_train_batch_size=1` with gradient accumulation of 4 (effective batch size = 4)
4. **Minimal Epochs**: Only 1 epoch for full fine-tuning, 3 for PEFT (but limited by max_steps=1)
5. **Small Evaluation Set**: Only 10 samples for ROUGE evaluation (high variance)

---

## Comprehensive Improvement Strategies

### 1. **Training Duration & Steps** (HIGHEST IMPACT)

#### Current Issues:
- `max_steps=1` prevents any meaningful learning
- Only 1 epoch for full fine-tuning

#### Recommended Changes:
```python
# For Full Fine-tuning (with local GPU)
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    num_train_epochs=3,              # Increase from 1 to 3-5 epochs
    weight_decay=0.01,
    logging_steps=50,                # Increase from 1 for cleaner logs
    max_steps=-1,                    # REMOVE or set to -1 (use epochs instead)
    per_device_train_batch_size=2,   # Increase if GPU memory allows (4-8)
    gradient_accumulation_steps=8,   # Increase to 8-16 for effective batch size
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    save_steps=500,                  # Save checkpoints periodically
    evaluation_strategy="steps",     # Evaluate during training
    eval_steps=500,                  # Evaluate every 500 steps
    save_total_limit=3,              # Keep only 3 best checkpoints
    load_best_model_at_end=True,     # Load best checkpoint at end
)

# For PEFT/LoRA (more efficient, can train longer)
peft_training_args = TrainingArguments(
    output_dir="./peft-results",
    per_device_train_batch_size=4,   # PEFT uses less memory, can be higher
    gradient_accumulation_steps=8,   # Effective batch size = 32
    num_train_epochs=5,              # Can train longer with PEFT (5-10 epochs)
    learning_rate=1e-4,              # Slightly higher for PEFT
    logging_steps=50,
    max_steps=-1,                    # REMOVE the limit
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    warmup_steps=100,                # Add warmup for better convergence
)
```

**Expected Impact**: 🔥🔥🔥 **CRITICAL** - This alone could improve ROUGE scores by 50-200%

---

### 2. **Data Utilization** (VERY HIGH IMPACT)

#### Current Issues:
- Only using 2% of available data (every 50th example)
- DialogSum has ~13,000 training examples; you're using ~260

#### Recommended Changes:
```python
# Option 1: Use ALL training data (recommended if you have GPU time)
# tokenized_datasets = tokenized_datasets  # Don't filter at all

# Option 2: Use more data (10-25% instead of 2%)
tokenized_datasets = tokenized_datasets.filter(lambda example, idx: idx % 10 == 0, with_indices=True)  # 10% of data

# Option 3: Use ALL data for PEFT (it's more efficient)
# For full fine-tuning, you might still subsample if training is too slow
```

**Expected Impact**: 🔥🔥🔥 **CRITICAL** - Could improve scores by 100-300%

---

### 3. **Learning Rate Optimization**

#### Current Issues:
- Same learning rate for all scenarios
- No learning rate scheduling

#### Recommended Changes:
```python
# Add learning rate scheduler
from transformers import get_linear_schedule_with_warmup

# For full fine-tuning
training_args = TrainingArguments(
    learning_rate=2e-5,              # Good starting point
    lr_scheduler_type="cosine",      # Or "linear" with warmup
    warmup_ratio=0.1,                # Warmup for 10% of training
    # ... other args
)

# For PEFT/LoRA (can use higher LR)
peft_training_args = TrainingArguments(
    learning_rate=1e-4,              # 5x higher for LoRA
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,               # Shorter warmup for PEFT
    # ... other args
)

# Experiment with these learning rates:
# Full fine-tuning: 1e-5, 2e-5, 3e-5, 5e-5
# PEFT: 5e-5, 1e-4, 2e-4, 3e-4
```

**Expected Impact**: 🔥🔥 **HIGH** - Could improve scores by 10-30%

---

### 4. **Batch Size & Gradient Accumulation** (GPU-Specific)

#### Current Issues:
- Very small effective batch size (4)
- Not utilizing GPU capacity

#### Recommended Changes:
```python
# First, find your GPU's maximum batch size:
def find_max_batch_size(model, tokenized_dataset, start_size=1, max_size=32):
    """Binary search to find maximum batch size that fits in GPU memory"""
    for batch_size in [1, 2, 4, 8, 16, 32]:
        if batch_size > max_size:
            break
        try:
            print(f"Testing batch_size={batch_size}...")
            test_args = TrainingArguments(
                output_dir="./temp",
                per_device_train_batch_size=batch_size,
                max_steps=2,
                logging_steps=1,
            )
            test_trainer = Trainer(
                model=model,
                args=test_args,
                train_dataset=tokenized_dataset["train"].select(range(min(100, len(tokenized_dataset["train"])))),
            )
            test_trainer.train()
            print(f"✓ batch_size={batch_size} works!")
            torch.cuda.empty_cache()
            max_working = batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"✗ batch_size={batch_size} - OOM")
                torch.cuda.empty_cache()
                break
            raise e
    return max_working

# Once you know your max batch size, aim for effective batch size of 32-64:
# Effective Batch Size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus

# Example: If max batch size is 4:
per_device_train_batch_size = 4
gradient_accumulation_steps = 8  # Effective = 32

# Example: If max batch size is 8:
per_device_train_batch_size = 8
gradient_accumulation_steps = 4  # Effective = 32
```

**Expected Impact**: 🔥 **MEDIUM** - Could improve scores by 5-15%

---

### 5. **LoRA Hyperparameter Optimization** (PEFT-Specific)

#### Current Issues:
- Conservative LoRA configuration (r=16, limited target modules)

#### Recommended Changes:
```python
# More aggressive LoRA configuration for better performance
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=32,                            # Increase from 16 to 32 or 64
    lora_alpha=64,                   # Typically 2x the rank
    lora_dropout=0.05,               # Reduce dropout (was 0.1)
    target_modules=["q", "v", "k", "o", "wi", "wo"],  # Add more modules
    bias="none",
)

# Alternative configurations to try:
# Configuration 1: Higher capacity
r=64, lora_alpha=128, target_modules=["q", "v", "k", "o", "wi", "wo"]

# Configuration 2: Maximum coverage
r=32, lora_alpha=64, target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]

# Configuration 3: Balanced
r=32, lora_alpha=32, lora_dropout=0.1, target_modules=["q", "v", "k", "o"]
```

**Expected Impact**: 🔥🔥 **HIGH** - Could improve PEFT scores by 20-50%

---

### 6. **Generation Configuration**

#### Current Issues:
- Default generation parameters may not be optimal

#### Recommended Changes:
```python
# Create a custom generation config for better summaries
from transformers import GenerationConfig

generation_config = GenerationConfig(
    max_new_tokens=128,              # Adjust based on average summary length
    min_new_tokens=20,               # Ensure minimum length
    num_beams=4,                     # Use beam search instead of greedy
    length_penalty=0.6,              # Slightly favor longer sequences
    no_repeat_ngram_size=3,          # Prevent repetition
    early_stopping=True,             # Stop when all beams finish
    temperature=1.0,                 # Adjust for more/less diversity (0.7-1.0)
    do_sample=False,                 # Deterministic with beam search
)

# Use during generation:
outputs = model.generate(**inputs, generation_config=generation_config)

# Alternative: Sampling-based generation
generation_config_sampling = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=20,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
)
```

**Expected Impact**: 🔥 **MEDIUM** - Could improve scores by 10-20%

---

### 7. **Evaluation Improvements**

#### Current Issues:
- Only 10 samples for evaluation (high variance)
- Single random seed

#### Recommended Changes:
```python
# Use larger, more representative evaluation set
np.random.seed(42)
random_indices = np.random.choice(len(dataset["test"]), size=100, replace=False)  # 100 instead of 10
samples = [dataset["test"][i] for i in random_indices]

# Add additional metrics beyond ROUGE
from evaluate import load

rouge = load("rouge")
bleu = load("bleu")
meteor = load("meteor")
bertscore = load("bertscore")

# Compute multiple metrics
rouge_scores = rouge.compute(predictions=predictions, references=references)
bleu_scores = bleu.compute(predictions=predictions, references=references)
meteor_scores = meteor.compute(predictions=predictions, references=references)
bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")

# Also compute per-example scores to identify problem cases
for i, (ref, pred) in enumerate(zip(references, predictions)):
    score = rouge.compute(predictions=[pred], references=[ref])
    print(f"Example {i}: ROUGE-L = {score['rougeL']:.3f}")
```

**Expected Impact**: Better measurement accuracy, helps identify which improvements work

---

### 8. **Data Preprocessing Enhancements**

#### Recommended Changes:
```python
# Add input length filtering
def filter_by_length(example):
    """Filter out extremely short or long dialogues"""
    dialogue_length = len(example["dialogue"].split())
    summary_length = len(example["summary"].split())
    return 20 < dialogue_length < 500 and 5 < summary_length < 100

dataset = dataset.filter(filter_by_length)

# Add data shuffling
tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42)

# Analyze dataset for quality
def analyze_dataset(dataset_split):
    """Print dataset statistics"""
    dialogue_lengths = [len(ex["dialogue"].split()) for ex in dataset_split]
    summary_lengths = [len(ex["summary"].split()) for ex in dataset_split]
    
    print(f"Dialogue length - Mean: {np.mean(dialogue_lengths):.1f}, Std: {np.std(dialogue_lengths):.1f}")
    print(f"Summary length - Mean: {np.mean(summary_lengths):.1f}, Std: {np.std(summary_lengths):.1f}")
    print(f"Compression ratio: {np.mean(dialogue_lengths) / np.mean(summary_lengths):.2f}x")

analyze_dataset(dataset["train"])
```

**Expected Impact**: 🔥 **MEDIUM** - Could improve scores by 5-10%

---

### 9. **GPU Memory Optimization**

```python
# Enable more aggressive memory optimizations
import torch

# Clear cache before training
torch.cuda.empty_cache()

# Enable TF32 for faster training on Ampere GPUs (RTX 30xx, A100, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use gradient checkpointing with more aggressive settings
model.gradient_checkpointing_enable()

# Monitor GPU memory
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")

print_gpu_memory()
```

**Expected Impact**: Enables larger batch sizes and longer training

---

### 10. **Advanced Techniques**

```python
# 1. Mixed Precision Training (already enabled)
# Ensure you're using bf16 if supported, otherwise fp16

# 2. Gradient Clipping
training_args = TrainingArguments(
    max_grad_norm=1.0,  # Prevent gradient explosion
    # ... other args
)

# 3. Weight Decay
training_args = TrainingArguments(
    weight_decay=0.01,  # L2 regularization
    # ... other args
)

# 4. Label Smoothing
training_args = TrainingArguments(
    label_smoothing_factor=0.1,  # Prevent overconfidence
    # ... other args
)

# 5. Early Stopping (using callbacks)
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

---

## Recommended Action Plan

### **Phase 1: Quick Wins (1-2 hours training time)**

1. ✅ Remove `max_steps=1` limitation → Set `max_steps=-1`
2. ✅ Increase epochs to 3 for full fine-tuning, 5 for PEFT
3. ✅ Use 10-25% of data instead of 2%
4. ✅ Increase evaluation samples to 50-100

**Expected ROUGE-L improvement**: From ~0.16-0.19 → **0.25-0.35**

### **Phase 2: Medium Optimizations (3-5 hours training time)**

5. ✅ Use ALL training data (remove filtering)
6. ✅ Optimize LoRA config (r=32, more target modules)
7. ✅ Increase batch size and gradient accumulation
8. ✅ Add beam search for generation

**Expected ROUGE-L improvement**: From 0.25-0.35 → **0.35-0.45**

### **Phase 3: Advanced Fine-tuning (5-10 hours training time)**

9. ✅ Train for 5-10 epochs with early stopping
10. ✅ Hyperparameter tuning (learning rate, warmup, etc.)
11. ✅ Try multiple LoRA configurations
12. ✅ Ensemble or checkpoint averaging

**Expected ROUGE-L improvement**: From 0.35-0.45 → **0.40-0.50+**

---

## Benchmark Target Scores

For DialogSum with FLAN-T5-base, well-trained models typically achieve:

- **ROUGE-1**: 0.45-0.55
- **ROUGE-2**: 0.15-0.25
- **ROUGE-L**: 0.35-0.45
- **ROUGE-Lsum**: 0.35-0.45

Your current best (fine-tuned): ROUGE-L = 0.189
**Target improvement**: 2-2.5x current scores

---

## GPU-Specific Recommendations

### For Different GPU Types:

**RTX 3060/3070 (8-12GB):**
- Batch size: 2-4
- Gradient accumulation: 8-16
- Use ALL data with gradient checkpointing
- Train PEFT with r=32

**RTX 3080/3090 (10-24GB):**
- Batch size: 4-8
- Gradient accumulation: 4-8
- Can use larger LoRA rank (r=64)
- Can train full fine-tuning efficiently

**RTX 4070/4080/4090 or A100:**
- Batch size: 8-16
- Gradient accumulation: 2-4
- Maximum LoRA configuration
- Can train even larger models

---

## Monitoring Training Progress

```python
# Add TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

training_args = TrainingArguments(
    logging_dir="./logs",
    logging_steps=10,
    report_to="tensorboard",
    # ... other args
)

# View in terminal: tensorboard --logdir=./logs
# Or in notebook:
%load_ext tensorboard
%tensorboard --logdir logs
```

---

## Expected Timeline

- **Phase 1 (Quick Wins)**: 1-3 hours total training
- **Phase 2 (Medium Opts)**: 3-8 hours total training  
- **Phase 3 (Advanced)**: 5-15 hours total training

**Total time for best results**: 10-25 hours of GPU training (can be done overnight or over a weekend)

---

## Summary

The **single most important change** you can make: 

🔥 **Remove `max_steps=1` and train for at least 3-5 epochs with ALL or most of your data** 🔥

This alone will likely give you 10-20x improvement in scores. All other optimizations will then stack on top of this foundation.
