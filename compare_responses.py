# compare_responses.py - FIXED VERSION
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

class AdapterComparator:
    def __init__(self):
        self.BASE = "mistralai/Mistral-7B-Instruct-v0.2"
        self.CODE_ADAPTER = "ds_project/output/adapters"         # Adjust path if needed
        self.THEORY_ADAPTER = "ds_project/output_theory/adapters"

        self.tokenizer = None
        self.model = None

        self._load_model_and_adapters()

    def _load_model_and_adapters(self):
        print("📦 Loading base model + adapters...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            quantization_config=bnb_config
        )

        # Load code adapter first
        model = PeftModel.from_pretrained(base_model, self.CODE_ADAPTER, adapter_name="code")
        model.load_adapter(self.THEORY_ADAPTER, adapter_name="theory")

        self.model = model.eval()
        print("✅ Base + code + theory adapters loaded")

    def _build_prompt(self, user_input: str, mode: str) -> str:
        SYSTEM_PROMPTS = {
            "base": "You are a helpful Data Science Assistant.",
            "code": (
                "You are a helpful Data Science Assistant.\n"
                "Return exactly ONE executable Python code block for the current task."
            ),
            "theory": (
                "You are a helpful Data Science Assistant.\n"
                "Explain the concept clearly in prose."
            )
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[mode]},
            {"role": "user", "content": user_input}
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(prompt):].strip()

    def compare(self, user_input: str) -> dict:
        print("\n🧪 Comparing responses for:", user_input)

        # 1. Base model (disable adapters)
        self.model.disable_adapter_layers()
        prompt_base = self._build_prompt(user_input, mode="base")
        base_resp = self._generate(prompt_base)

        # 2. Code adapter
        self.model.set_adapter("code")
        prompt_code = self._build_prompt(user_input, mode="code")
        code_resp = self._generate(prompt_code)

        # 3. Theory adapter
        self.model.set_adapter("theory")
        prompt_theory = self._build_prompt(user_input, mode="theory")
        theory_resp = self._generate(prompt_theory)

        return {
            "user_input": user_input,
            "base_response": base_resp,
            "code_response": code_resp,
            "theory_response": theory_resp
        }

# Example usage
if __name__ == "__main__":
    comparator = AdapterComparator()

    user_input = "What is regularization in machine learning?"
    results = comparator.compare(user_input)

    print("\n===== BASE MODEL =====")
    print(results["base_response"])
    print("\n===== CODE ADAPTER =====")
    print(results["code_response"])
    print("\n===== THEORY ADAPTER =====")
    print(results["theory_response"])
