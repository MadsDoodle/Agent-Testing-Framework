import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_REGISTRY = {
    "qwen3b": {
        "model_id"    : "Qwen/Qwen2.5-3B-Instruct",
        "display_name": "Qwen 2.5 - 3B Instruct",
        "dtype"       : torch.float16,
        "load_in_4bit": False
    },
    "qwen1b": {
        "model_id"    : "Qwen/Qwen2.5-1.5B-Instruct",
        "display_name": "Qwen 2.5 - 1.5B Instruct (lightest)",
        "dtype"       : torch.float16,
        "load_in_4bit": False
    },
    "qwen7b": {
        "model_id"    : "Qwen/Qwen2.5-7B-Instruct",
        "display_name": "Qwen 2.5 - 7B Instruct (4-bit, fits T4)",
        "dtype"       : torch.float16,
        "load_in_4bit": True      # needs bitsandbytes
    },
    "smollm3": {
        "model_id"    : "HuggingFaceTB/SmolLM3-3B",
        "display_name": "SmolLM3 - 3B (HuggingFace)",
        "dtype"       : torch.float16,
        "load_in_4bit": False
    },
    "ministral3b": {
        "model_id"    : "ministral/Ministral-3b-instruct",
        "display_name": "Ministral 3B Instruct",
        "dtype"       : torch.float16,
        "load_in_4bit": False
    },
}

JUDGE_REGISTRY = {
    "same"       : None,          # reuse whatever agent model is loaded
    "qwen1b"     : "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b"     : "Qwen/Qwen2.5-3B-Instruct",
    "smollm3"    : "HuggingFaceTB/SmolLM3-3B",
    "ministral3b": "ministral/Ministral-3b-instruct",
}

# ── Change these two to control what runs ────────────────────────────────
MODEL_KEY  = "smollm3"    # agent model
JUDGE_KEY  = "same"       # "same" = reuse agent model, or pick a key from JUDGE_REGISTRY


def load_model(model_id: str, dtype=torch.float16, load_in_4bit=False):
    print(f"\n⏳ Loading: {model_id}")
    print(f"   4-bit: {load_in_4bit}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )

    mdl.eval()
    print(f"✅ Loaded on: {next(mdl.parameters()).device}")
    return mdl, tok


def get_judge(judge_key: str, agent_model, agent_tokenizer,
              agent_model_id: str, model_registry: dict):
    """
    Returns (judge_model, judge_tokenizer).
    If judge_key == "same" OR the judge model_id matches the already-loaded
    agent model_id, returns the agent model directly — no reload.
    """
    if judge_key == "same":
        print("⚖️  Judge = Agent model (reusing, no extra memory used)")
        return agent_model, agent_tokenizer

    judge_model_id = JUDGE_REGISTRY[judge_key]

    if judge_model_id == agent_model_id:
        print(f"⚖️  Judge model ID matches agent ({judge_model_id}) — reusing")
        return agent_model, agent_tokenizer

    # Different model — check if we have enough VRAM
    if torch.cuda.is_available():
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 1e9
        print(f"   Available GPU memory: {free_gb:.1f} GB")
        if free_gb < 3.5:
            print("⚠️  Less than 3.5GB free — falling back to agent model as judge")
            return agent_model, agent_tokenizer

    # Check if this judge_key is in MODEL_REGISTRY for 4-bit config
    if judge_key in model_registry:
        jcfg = model_registry[judge_key]
        jmodel, jtok = load_model(jcfg["model_id"], jcfg["dtype"], jcfg["load_in_4bit"])
    else:
        jmodel, jtok = load_model(judge_model_id)

    return jmodel, jtok