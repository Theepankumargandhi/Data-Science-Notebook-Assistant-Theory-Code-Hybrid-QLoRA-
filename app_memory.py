# app_memory.py
import re
import gradio as gr
import torch
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------- Model + adapter paths ----------------
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
CODE_ADAPTER_DIR = "output/adapters"             # code LoRA
THEORY_ADAPTER_DIR = "output_theory/adapters"    # theory LoRA

# ---------------- System prompts ----------------
SYSTEM_BASE = (
    "You are a helpful Data Science Notebook Assistant.\n"
    "Do not repeat the system message or instructions in your reply.\n"
    "Be concise and correct for Python/Pandas/Matplotlib/scikit-learn tasks."
)
SYSTEM_CODE = (
    SYSTEM_BASE + "\n"
    "Return exactly ONE executable Python code block for the current task, with minimal comments and no extra prose.\n"
    "Answer ONLY the current question."
)
SYSTEM_EXPLAIN = (
    SYSTEM_BASE + "\n"
    "Answer in clear prose. Include code only if essential to explain the concept.\n"
    "Answer ONLY the current question."
)

# ---------------- Cleaners ----------------
INSTRUCTION_LINE_PATTERNS = [
    r"^\s*(you\s+are\s+a\s+helpful\s+data\s+science\s+notebook\s+assistant.*)$",
    r"^\s*do\s+not\s+repeat.*$",
    r"^\s*be\s+concise.*$",
    r"^\s*answer\s+only\s+the\s+current\s+question.*$",
    r"^\s*return\s+exactly\s+one\s+executable\s+python\s+code\s+block.*$",
    r"^\s*single\s+executable\s+python\s+code\s+block.*$",
    r"^\s*no\s+extra\s+prose.*$",
    r"^\s*answer\s+in\s+clear\s+prose.*$",
    r"^\s*include\s+code\s+only\s+if\s+essential.*$",
]
INSTRUCTION_RX = [re.compile(p, re.I) for p in INSTRUCTION_LINE_PATTERNS]

def _strip_inst(text: str) -> str:
    return text.replace("[INST]", "").replace("[/INST]", "").strip()

def _strip_instruction_lines(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        l = line.rstrip("\n")
        if any(rx.match(l) for rx in INSTRUCTION_RX):
            continue
        cleaned.append(l)
    return "\n".join(cleaned).strip()

def _extract_fenced_code(text: str):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if not m:
        return None
    body = m.group(1).strip()
    body_no_instr = _strip_instruction_lines(body)
    if not body_no_instr or len(body_no_instr.split()) < 2:
        return None
    return body_no_instr

CODE_LIKE_PREFIXES = (
    "import ", "from ", "plt.", "sns.", "pd.", "np.", "df.", "torch.",
    "sklearn", "X_train", "X_test", "y_train", "y_test"
)

def _heuristic_code_only(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    keep = []
    for ln in lines:
        ll = ln.lstrip()
        if not ll:
            keep.append(ln)
            continue
        if ll.startswith(CODE_LIKE_PREFIXES):
            keep.append(ln)
            continue
        if any(sym in ll for sym in ("=", "(", ")", "[", "]", "{", "}", ":", ".", ",")):
            keep.append(ln)
    code = "\n".join([l for l in keep if l.strip()])
    return f"```python\n{code}\n```" if code else text.strip()

def _deduplicate_sentences(text: str) -> str:
    t = " ".join(text.split())
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    seen, out = set(), []
    for s in parts:
        key = s.lower()
        if key not in seen:
            seen.add(key); out.append(s)
    return " ".join(out)

def _postprocess_output(raw: str, mode: str) -> str:
    raw = _strip_inst(raw)
    raw = _strip_instruction_lines(raw)
    code = _extract_fenced_code(raw)
    if code is not None:
        return f"```python\n{code}\n```"
    if mode == "code":
        raw = _strip_instruction_lines(raw)
        return _heuristic_code_only(raw)
    return _deduplicate_sentences(raw.strip())

def _clean_for_history(text: str, mode: str) -> str:
    return _postprocess_output(text, mode)

# ---------------- Router / guardrail ----------------
CODE_HINTS = (
    "write code,python code,script,plot,draw,visualize,chart,histogram,scatter,bar,line,box,heatmap,"
    "pandas,groupby,merge,join,filter,select,loc,iloc,rename,drop,fillna,read_csv,describe,sklearn,"
    "train,fit,model,logistic,randomforest,regression,accuracy,roc,auc,confusion"
).split(",")
EXPLAIN_HINTS = (
    "explain,define,what is,why,compare,difference,intuition,pros cons,when to use,theory,concept"
).split(",")

def _route_mode(user_text: str) -> str:
    t = user_text.lower()
    if any(k in t for k in EXPLAIN_HINTS): return "explain"
    if any(k in t for k in CODE_HINTS):    return "code"
    if re.match(r"^\s*(what|why|how|when|define)\b", t): return "explain"
    return "code"

# ---------------- DS guardrail (add NLP + better matching) ----------------
DS_TERMS = [
    # general
    "data science","machine learning","ml","ai","artificial intelligence","nlp","natural language processing",
    "statistics","stat","eda","feature engineering",
    # python libs
    "python","pandas","numpy","matplotlib","seaborn","scikit","sklearn","dataframe","df","csv",
    # tasks & eval
    "regression","classification","clustering","time series","train","test","split","cross validation",
    "k-fold","roc","auc","precision","recall","f1","rmse","mae","accuracy","confusion matrix",
    # models
    "random forest","randomforest","decision tree","decision trees","tree-based",
    "gradient boosting","xgboost","lightgbm","catboost",
    "linear regression","logistic regression","ridge","lasso",
    "svm","support vector machine","knn","k-nn","k nearest neighbors",
    "naive bayes","bayes",
    # common theory
    "overfitting","underfitting","bias variance","bias-variance","regularization","dropout",
    "feature selection","feature scaling","standardization","normalization","one-hot encoding",
    "imputation","missing data","stratified","bootstrapping","confidence interval","p-value",
    # interpretability, drift, nlp
    "shap","lime","partial dependence","permutation importance","feature importance",
    "concept drift","data drift","tokenization","embedding","language model","transformer","bert","gpt"
]

def _is_ds_query(text: str) -> bool:
    t = text.lower()
    t_compact = t.replace(" ", "").replace("-", "")
    for k in DS_TERMS:
        k_c = k.replace(" ", "").replace("-", "")
        if k in t or k_c in t_compact:
            return True
    return False


NEEDS_CONTEXT_PAT = re.compile(
    r"\b(now|then|continue|next|also|as above|same df|use the same|based on (it|that|above)|"
    r"with the previous|using earlier code|continue from|from before|that|it|those)\b",
    flags=re.I,
)
def _needs_context(user_text: str) -> bool:
    return bool(NEEDS_CONTEXT_PAT.search(user_text or ""))

# ---------------- Load model + adapters ----------------
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model_base = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    quantization_config=bnb,
)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model_base.config.pad_token_id = tok.pad_token_id

HAS_ENABLE_DISABLE = False
CODE_ADAPTER_NAME = "code"
THEORY_ADAPTER_NAME = "theory"

try:
    model = PeftModel.from_pretrained(model_base, CODE_ADAPTER_DIR, adapter_name=CODE_ADAPTER_NAME)
    model.load_adapter(THEORY_ADAPTER_DIR, adapter_name=THEORY_ADAPTER_NAME)
    HAS_ENABLE_DISABLE = hasattr(model, "disable_adapter_layers") and hasattr(model, "enable_adapter_layers")
except TypeError:
    model = PeftModel.from_pretrained(model_base, CODE_ADAPTER_DIR)
    try: CODE_ADAPTER_NAME = next(iter(model.peft_config.keys()))
    except Exception: CODE_ADAPTER_NAME = "default"
    model.load_adapter(THEORY_ADAPTER_DIR, adapter_name=THEORY_ADAPTER_NAME)
    HAS_ENABLE_DISABLE = hasattr(model, "disable_adapter_layers") and hasattr(model, "enable_adapter_layers")

try: model.set_adapter(CODE_ADAPTER_NAME)
except Exception: pass

# ---------------- Status chip (HTML, no emojis) ----------------
def _chip_html(kind: str) -> str:
    label = {
        "code": "Adapter: CODE",
        "theory": "Adapter: THEORY",
        "base": "Adapter: BASE",
        "guard": "Guardrail: BLOCKED",
    }.get(kind, "Adapter: UNKNOWN")
    cls = {
        "code": "chip code",
        "theory": "chip theory",
        "base": "chip base",
        "guard": "chip guard",
    }.get(kind, "chip")
    return f'<div class="{cls}">{label}</div>'

# ---------------- Message building ----------------
def _select_system(mode: str) -> str:
    if mode == "code":    return SYSTEM_CODE
    if mode == "explain": return SYSTEM_EXPLAIN
    return SYSTEM_BASE

def _build_messages(history: List[Tuple[str, str]], user_text: str, mode: str, max_hist_turns: int) -> list:
    messages = [{"role": "system", "content": _select_system(mode)}]
    if _needs_context(user_text):
        trimmed = history[-max_hist_turns:] if max_hist_turns > 0 else []
        for u, a in trimmed:
            u_clean = _strip_instruction_lines(_strip_inst(u or ""))
            a_clean = _clean_for_history(a or "", mode)
            if u_clean: messages.append({"role": "user", "content": u_clean})
            if a_clean: messages.append({"role": "assistant", "content": a_clean})
    messages.append({"role": "user", "content": user_text})
    return messages

# ---------------- Adapter selection ----------------
def _select_adapter_for_request(is_ds: bool, routed_mode: str, ds_guard: bool):
    state = {"disabled": False, "active": None}
    if ds_guard and not is_ds:
        return "guard", state
    if not ds_guard and not is_ds:
        if HAS_ENABLE_DISABLE:
            try:
                model.disable_adapter_layers(); state["disabled"] = True
            except Exception: pass
        return "base", state
    if HAS_ENABLE_DISABLE:
        try: model.enable_adapter_layers()
        except Exception: pass
    name = THEORY_ADAPTER_NAME if routed_mode == "explain" else CODE_ADAPTER_NAME
    try: model.set_adapter(name); state["active"] = name
    except Exception: pass
    return ("theory" if routed_mode == "explain" else "code"), state

def _restore_after_request(state: dict):
    if state.get("disabled") and HAS_ENABLE_DISABLE:
        try: model.enable_adapter_layers()
        except Exception: pass

# ---------------- Generation ----------------
def _generate_reply(messages: list, mode: str) -> str:
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=300, temperature=0.2, top_p=0.9, repetition_penalty=1.1,
        )
    raw = tok.decode(out[0], skip_special_tokens=True)
    return _postprocess_output(raw, mode)

# ---------------- Custom CSS (clean, professional) ----------------
CUSTOM_CSS = """
/* Layout tweaks */
.gradio-container {max-width: 1080px !important; margin: 0 auto;}
footer {display:none;}

/* Header */
.app-title h2 {margin: 0 0 8px 0; font-weight: 700;}
.app-subtitle {color:#667085; font-size: 0.95rem;}

/* Sidebar controls */
.svelte-1ipelgc, .svelte-1h6o1oe {background: transparent;}
/* Chatbot */
.gradio-chatbot .message {font-size: 15px; line-height: 1.5;}
/* Code blocks */
pre code {font-size: 14px; line-height: 1.55;}
/* Buttons */
button {border-radius: 12px;}
/* Status chip */
.chip {display:inline-block; padding:6px 12px; border-radius:9999px; font-weight:600; font-size:12px; letter-spacing:0.02em;}
.chip.code   {background:#e8f1ff; color:#0b57d0; border:1px solid #bcd3ff;}
.chip.theory {background:#eaf6ef; color:#0a7a46; border:1px solid #bfe3cf;}
.chip.base   {background:#f2f4f7; color:#344054; border:1px solid #e4e7ec;}
.chip.guard  {background:#fff3f2; color:#b42318; border:1px solid #ffd1cc;}
/* Chip container */
#chip-holder {margin: 8px 0 2px 2px;}
"""

# ---------------- UI ----------------
with gr.Blocks(title="DS Notebook Assistant — Theory + Code Hybrid (QLoRA)", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "<div class='app-title'>"
        "<h2>Data Science Notebook Assistant — Theory + Code Hybrid (QLoRA)</h2>"
        "</div>"
    )

    with gr.Row():
        chatbot = gr.Chatbot(height=430, label="Chat")
        with gr.Column(scale=0.42):
            mode = gr.Radio(
                choices=["Auto", "Code", "Explain"],
                value="Auto",
                label="Response style",
                info="Auto routes by intent; Code returns one Python block; Explain returns prose."
            )
            ds_only = gr.Checkbox(
                value=True, label="Enforce DS-only guardrail",
                info="When on, only DS questions are answered (non-DS are politely declined)."
            )
            max_hist = gr.Slider(2, 12, value=6, step=1, label="History window (used only when you ask to continue)",
                                 info="History is included only if you explicitly refer to prior turns.")
            clear_btn = gr.Button("Clear chat")
    chip = gr.HTML(_chip_html("code"), elem_id="chip-holder")

    user = gr.Textbox(placeholder="Ask something (e.g., 'Plot histogram of price' or 'What is overfitting?')", label="Message")
    send = gr.Button("Send", variant="primary")
    state = gr.State([])  # list of (user, assistant) tuples

    def respond(user_text, history, mode_sel, ds_guard, max_h):
        if not user_text or not user_text.strip():
            return gr.update(), history, chip.value

        is_ds = _is_ds_query(user_text)
        routed = _route_mode(user_text) if mode_sel == "Auto" else ("code" if mode_sel.lower() == "code" else "explain")

        chip_kind, sel_state = _select_adapter_for_request(is_ds, routed, ds_guard)
        if chip_kind == "guard":
            reply = ("I focus on Python / Data Science tasks (Pandas, Matplotlib, scikit-learn). "
                     "Please rephrase your request in that domain.")
            new_hist = history + [(user_text, reply)]
            return new_hist, new_hist, _chip_html("guard")

        N = int(max_h) if isinstance(max_h, (int, float)) else 6
        messages = _build_messages(history, user_text, routed, N)

        reply = _generate_reply(messages, routed)
        _restore_after_request(sel_state)

        new_hist = history + [(user_text, reply)]
        return new_hist, new_hist, _chip_html(chip_kind)

    def clear_chat():
        return [], [], _chip_html("code")

    send.click(fn=respond, inputs=[user, state, mode, ds_only, max_hist], outputs=[chatbot, state, chip])
    user.submit(fn=respond, inputs=[user, state, mode, ds_only, max_hist], outputs=[chatbot, state, chip])
    clear_btn.click(fn=clear_chat, inputs=None, outputs=[chatbot, state, chip])

demo.launch()
