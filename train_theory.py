# train_theory.py
import os, inspect, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------------
# Paths / config
# -------------------
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_FILE = "data/train_theory.jsonl"          # <-- theory dataset
OUT_DIR   = "output_theory"                    # <-- separate folder so you don't overwrite code adapter
MAX_LEN   = 1024
EPOCHS    = 2
BATCH_SZ  = 1
GRAD_ACC  = 16
LR        = 2e-4
LORA_R    = 16
LORA_ALPHA= 16
LORA_DROPOUT = 0.05

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------
# Tokenizer & 4-bit base (QLoRA)
# -------------------
print("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading 4-bit base model…")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    quantization_config=bnb,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

# -------------------
# LoRA adapters on attention+MLP
# (same targets as code adapter so both are compatible at inference)
# -------------------
model = prepare_model_for_kbit_training(model)
lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# -------------------
# Data: load single JSONL, split 90/10
# expects: {"messages":[{"role":"system","content":"..."}, {"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
# -------------------
print("Loading dataset…")
ds_all = load_dataset("json", data_files=DATA_FILE, split="train")
splits = ds_all.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = splits["train"], splits["test"]

def to_prompt(example):
    txt = tok.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": txt}

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=MAX_LEN)

print("Preprocessing…")
train_ds = train_ds.map(to_prompt, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(to_prompt,  remove_columns=eval_ds.column_names)
train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
eval_ds  = eval_ds.map(tokenize,  batched=True, remove_columns=["text"])

dc = DataCollatorForLanguageModeling(tok, mlm=False)

# -------------------
# TrainingArguments with eval arg fallback
# -------------------
args_kwargs = dict(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SZ,
    per_device_eval_batch_size=BATCH_SZ,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    save_strategy="steps",
    save_steps=200,
    logging_steps=50,
    gradient_checkpointing=True,
    bf16=torch.cuda.is_available(),
    fp16=not torch.cuda.is_available(),
    report_to="none",
)

sig = inspect.signature(TrainingArguments)
if "evaluation_strategy" in sig.parameters:
    args_kwargs["evaluation_strategy"] = "steps"
    args_kwargs["eval_steps"] = 100
else:
    args_kwargs["eval_strategy"] = "steps"
    args_kwargs["eval_steps"] = 100

training_args = TrainingArguments(**args_kwargs)

# -------------------
# Train
# -------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=dc,
)

print("Starting training…")
trainer.train()

# -------------------
# Save adapters
# -------------------
adp_dir = os.path.join(OUT_DIR, "adapters")  # e.g., output_theory/adapters
model.save_pretrained(adp_dir)
tok.save_pretrained(adp_dir)
print(f"✅ Saved Theory LoRA adapters to {adp_dir}")
