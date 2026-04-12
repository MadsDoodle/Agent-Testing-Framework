import os

CFG = {
    "model_key"       : "qwen3b",
    "results_dir"     : "evaluation/results",
    "data_dir"        : "data",
    "rag_docs_dir"    : "data/rag_docs",
    "embed_model"     : "all-MiniLM-L6-v2",
    "chunk_size"      : 300,
    "chunk_overlap"   : 50,
    "top_k_retrieval" : 3,
    "min_resp_len"    : 20,
    "judge_max_tokens": 256,
    "agent_max_tokens": 300,
}

for d in [CFG["results_dir"], CFG["data_dir"], CFG["rag_docs_dir"]]:
    os.makedirs(d, exist_ok=True)