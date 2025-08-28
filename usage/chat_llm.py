# chat_llm.py
# -*- coding: utf-8 -*-
"""
Unified helper for InferenceVision LLMs.
"""

import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer

_state = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "lock": threading.Lock(),
    "current_name": None,
}

MODEL_MAP = {
    "inferencevision-pythia-1B": {
        "repo": "doguilmak/inferencevision-pythia-1B",
    },
    "inferencevision-gpt-neo-1.3B": {
        "repo": "doguilmak/inferencevision-gpt-neo-1.3B",
    },
}

def load_model(model_name: str):
    """Load tokenizer & model (thread-safe)."""
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_MAP.keys())}")

    if _state["model"] is not None and _state["current_name"] == model_name:
        return

    with _state["lock"]:
        if _state["model"] is not None and _state["current_name"] == model_name:
            return

        info = MODEL_MAP[model_name]
        repo_id = info["repo"]

        print(f"[chat_llm] Loading model '{model_name}' from {repo_id} ...")

        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=None)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, trust_remote_code=True, token=None
        )

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        _state["model"] = model
        _state["tokenizer"] = tokenizer
        _state["device"] = device
        _state["current_name"] = model_name

        print(f"[chat_llm] Model '{model_name}' loaded on {device}")

def _build_prompt(q: str, short: bool) -> str:
    return f"Q: {q}\nA (concise, 1-2 sentences):" if short else f"Q: {q}\nA:"

def ask(text: str, short: bool = False, max_new_tokens: int = 128, do_sample: bool = False) -> str:
    """Ask the current loaded model a question."""
    if not text or _state["model"] is None:
        raise RuntimeError("No model loaded. Call load_model(name) first.")

    tokenizer = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]

    prompt = _build_prompt(text.strip(), short)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prefix = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    reply = full[len(prefix):].strip() if full.startswith(prefix) else full.strip()
    return reply

def loop():
    """Simple blocking REPL for Q&A."""
    if _state["model"] is None:
        raise RuntimeError("No model loaded. Call load_model(name) first.")

    print("InferenceVision loop. Type '/exit' or '/quit' to stop.")
    print("Use '/short on' or '/short off' to toggle concise answers.")

    short_mode = False
    while True:
        try:
            txt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Bye!")
            break
        if not txt:
            continue
        low = txt.lower()
        if low in ("/exit", "/quit"):
            print("Exiting. Bye!")
            break
        if low.startswith("/short"):
            parts = low.split()
            if len(parts) > 1 and parts[1] in ("on", "off"):
                short_mode = parts[1] == "on"
                print(f"[short mode -> {short_mode}]")
            else:
                print(f"[short mode is {short_mode}]")
            continue

        try:
            ans = ask(txt, short=short_mode)
            print("\nModel:", ans)
        except Exception as e:
            print("[error]", e)


# ------------------- Usage -------------------
# from chat_llm import load_model, ask, loop

# # Load your Pythia model from subfolder
# load_model("inferencevision-pythia-1b")
# print(ask("What is machine learning?"))

# # Switch to GPT-Neo model
# load_model("inferencevision-gpt-neo-1.3B")
# print(ask("Summarize quantum mechanics in 2 sentences.", short=True))

# # Interactive loop
# loop()
