import time
import requests
import torch
from abc import ABC, abstractmethod


class AgentInterface(ABC):
    @abstractmethod
    def run(self, input_text: str, context: str = "") -> dict:
        pass


# ── Local HF Model Agent (from previous stage, unchanged) ──────────
class LLMAgent(AgentInterface):
    SYSTEM_PROMPT = (
        "You are a helpful, honest, and safe AI assistant. "
        "Answer clearly and concisely. "
        "If the request is harmful, illegal, or unethical, refuse politely."
    )

    def __init__(self, model, tokenizer, model_key, max_new_tokens=300):
        self.model          = model
        self.tokenizer      = tokenizer
        self.model_key      = model_key
        self.max_new_tokens = max_new_tokens

    def run(self, input_text: str, context: str = "") -> dict:
        user_content = f"Context:\n{context}\n\nQuestion: {input_text}" if context else input_text
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_content}
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"System: {self.SYSTEM_PROMPT}\nUser: {user_content}\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        start  = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        latency    = round(time.time() - start, 4)
        input_len  = inputs["input_ids"].shape[1]
        answer     = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        return {
            "answer"  : answer,
            "metadata": {"latency": latency, "tokens_used": int(outputs[0][input_len:].shape[0]), "retrieved_docs": None}
        }


# ── HTTP Agent — for any REST endpoint (OpenAI-compatible or HF) ────
class HTTPAgent(AgentInterface):
    """
    Supports:
    - openai     : OpenAI /v1/chat/completions format
    - huggingface: HuggingFace Inference API format
    - custom     : user-defined input_field/output_field
    """

    SYSTEM_PROMPT = (
        "You are a helpful, honest, and safe AI assistant. "
        "Answer clearly and concisely. "
        "If the request is harmful or unethical, refuse politely."
    )

    def __init__(self, url: str, api_format: str = "openai",
                 api_key: str = "", model_name: str = "gpt-3.5-turbo",
                 custom_input_field: str = "inputs",
                 custom_output_path: str = "generated_text"):
        self.url                = url
        self.api_format         = api_format
        self.model_name         = model_name
        self.custom_input_field = custom_input_field
        self.custom_output_path = custom_output_path
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _build_payload(self, input_text: str, context: str) -> dict:
        user_content = f"Context:\n{context}\n\nQuestion: {input_text}" if context else input_text
        if self.api_format == "openai":
            return {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content}
                ],
                "max_tokens": 300,
                "temperature": 0
            }
        elif self.api_format == "huggingface":
            return {"inputs": user_content, "parameters": {"max_new_tokens": 300}}
        else:  # custom
            return {self.custom_input_field: user_content}

    def _parse_response(self, resp_json) -> str:
        try:
            # ── Auto-detect format regardless of what user selected ──────

            # OpenAI format — {"choices": [{"message": {"content": "..."}}]}
            if isinstance(resp_json, dict) and "choices" in resp_json:
                return resp_json["choices"][0]["message"]["content"].strip()

            # HuggingFace list format — [{"generated_text": "..."}]
            if isinstance(resp_json, list) and len(resp_json) > 0:
                item = resp_json[0]
                if "generated_text" in item:
                    return item["generated_text"].strip()
                # HF sometimes nests it
                if "text" in item:
                    return item["text"].strip()

            # HuggingFace dict format — {"generated_text": "..."}
            if isinstance(resp_json, dict) and "generated_text" in resp_json:
                return resp_json["generated_text"].strip()

            # Custom path — walk dot-separated keys
            if self.api_format == "custom":
                obj = resp_json
                for key in self.custom_output_path.split("."):
                    obj = obj[key]
                return str(obj).strip()

            # Last resort — dump whatever came back
            return f"[UNRECOGNIZED FORMAT] Raw: {str(resp_json)[:200]}"

        except Exception as e:
            return f"[PARSE ERROR: {e}] Raw: {str(resp_json)[:200]}"

    def run(self, input_text: str, context: str = "") -> dict:
        payload = self._build_payload(input_text, context)
        start   = time.time()
        try:
            resp    = requests.post(self.url, json=payload,
                                    headers=self.headers, timeout=30)
            resp.raise_for_status()
            answer  = self._parse_response(resp.json())
            status  = "success"
        except Exception as e:
            answer  = f"[HTTP ERROR: {e}]"
            status  = "failure"

        latency = round(time.time() - start, 4)
        return {
            "answer"  : answer,
            "metadata": {"latency": latency, "tokens_used": None,
                         "retrieved_docs": None, "status": status}
        }