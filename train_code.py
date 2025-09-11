#train_code.py
import os, inspect, torch
import mlflow
import mlflow.pytorch
import dvc.api
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from datetime import datetime

# -------------------
# MLflow Setup
# -------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ds-assistant-code-adapter")

# -------------------
# Paths / config
# -------------------
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_FILE = "data/train.jsonl"
OUT_DIR = "output"
DVC_MODEL_DIR = "models/code_adapter"
MAX_LEN = 1024
EPOCHS = 2
BATCH_SZ = 1
GRAD_ACC = 16
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Version tracking
VERSION = "v1.0"
EXPERIMENT_NAME = f"code_adapter_{VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DVC_MODEL_DIR, exist_ok=True)

# -------------------
# Start MLflow Run
# -------------------
with mlflow.start_run(run_name=EXPERIMENT_NAME):
    
    # Log parameters
    mlflow.log_params({
        "base_model": BASE,
        "max_length": MAX_LEN,
        "epochs": EPOCHS,
        "batch_size": BATCH_SZ,
        "gradient_accumulation_steps": GRAD_ACC,
        "learning_rate": LR,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "adapter_type": "code",
        "version": VERSION
    })
    
    # Log tags
    mlflow.set_tags({
        "adapter_type": "code",
        "version": VERSION,
        "model_family": "mistral-7b",
        "quantization": "4bit",
        "training_type": "qlora"
    })

    # -------------------
    # Tokenizer & 4-bit base (QLoRA)
    # -------------------
    print("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
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
        token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )

    # -------------------
    # LoRA adapters on attention+MLP
    # -------------------
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Log model architecture info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    mlflow.log_metrics({
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percentage": (trainable_params / total_params) * 100
    })

    # -------------------
    # Data: load single JSONL, split 90/10
    # -------------------
    print("Loading dataset…")
    ds_all = load_dataset("json", data_files=DATA_FILE, split="train")
    
    # Log dataset info
    mlflow.log_metrics({
        "total_examples": len(ds_all),
        "dataset_file": DATA_FILE
    })
    
    splits = ds_all.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = splits["train"], splits["test"]
    
    mlflow.log_metrics({
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds)
    })

    def to_prompt(example):
        txt = tok.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": txt}

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LEN)

    print("Preprocessing…")
    train_ds = train_ds.map(to_prompt, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(to_prompt, remove_columns=eval_ds.column_names)
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

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
        evaluation_strategy="steps",
        eval_steps=100,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    # Handle different transformers versions
    sig = inspect.signature(TrainingArguments)
    if "evaluation_strategy" not in sig.parameters:
        args_kwargs["eval_strategy"] = args_kwargs.pop("evaluation_strategy")

    training_args = TrainingArguments(**args_kwargs)

    # -------------------
    # Custom Trainer for MLflow logging
    # -------------------
    class MLflowTrainer(Trainer):
        def log(self, logs):
            super().log(logs)
            # Log metrics to MLflow
            step = logs.get("step", self.state.global_step)
            for key, value in logs.items():
                if isinstance(value, (int, float)) and key != "step":
                    mlflow.log_metric(key, value, step=step)

    # -------------------
    # Train
    # -------------------
    trainer = MLflowTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=dc,
    )

    print("Starting training…")
    train_result = trainer.train()

    # Log final training metrics
    mlflow.log_metrics({
        "final_train_loss": train_result.training_loss,
        "total_training_time": train_result.metrics.get("train_runtime", 0),
        "training_samples_per_second": train_result.metrics.get("train_samples_per_second", 0)
    })

    # -------------------
    # Save adapters (Local + DVC)
    # -------------------
    adp_dir = os.path.join(OUT_DIR, "adapters")
    model.save_pretrained(adp_dir)
    tok.save_pretrained(adp_dir)
    
    # Copy to DVC tracked directory
    import shutil
    dvc_adp_dir = os.path.join(DVC_MODEL_DIR, VERSION)
    if os.path.exists(dvc_adp_dir):
        shutil.rmtree(dvc_adp_dir)
    shutil.copytree(adp_dir, dvc_adp_dir)
    
    # Save training metadata for DVC
    metadata = {
        "version": VERSION,
        "model_type": "code_adapter",
        "base_model": BASE,
        "training_date": datetime.now().isoformat(),
        "mlflow_run_id": mlflow.active_run().info.run_id,
        "hyperparameters": {
            "epochs": EPOCHS,
            "learning_rate": LR,
            "lora_r": LORA_R,
            "batch_size": BATCH_SZ
        },
        "metrics": {
            "final_train_loss": train_result.training_loss,
            "trainable_params": trainable_params,
            "total_params": total_params
        }
    }
    
    with open(os.path.join(dvc_adp_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Log model artifact to MLflow
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="code_adapter",
        extra_files=[os.path.join(dvc_adp_dir, "metadata.json")]
    )
    
    # Log adapter files as artifacts
    mlflow.log_artifacts(dvc_adp_dir, "adapter_files")
    
    print(f"✅ Saved LoRA adapters to {adp_dir}")
    print(f"✅ Saved versioned adapters to DVC: {dvc_adp_dir}")
    print(f"✅ MLflow run ID: {mlflow.active_run().info.run_id}")
    
    # Model evaluation metrics (basic)
    eval_results = trainer.evaluate()
    mlflow.log_metrics({
        "final_eval_loss": eval_results["eval_loss"],
        "eval_perplexity": torch.exp(torch.tensor(eval_results["eval_loss"])).item()
    })
    
    print(f"✅ Final evaluation loss: {eval_results['eval_loss']:.4f}")
    print(f"✅ Training completed successfully!")