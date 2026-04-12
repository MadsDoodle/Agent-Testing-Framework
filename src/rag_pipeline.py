import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CFG


# ── Synthetic RAG Documents ─────────────────────────────────────────────────

DOCS = {

    # SIMPLE ─ plain text, no structure
    "return_policy": """
RETURN POLICY ─ TechStore Online

1. General Returns
Customers may return most items within 30 days of purchase for a full refund.
Items must be in original, unused condition and in original packaging.
Digital downloads and software licenses are non-refundable once activated.

2. Electronics
Electronics may be returned within 15 days if unopened.
Opened electronics are eligible for exchange only, not refund.
Damaged or defective electronics will be replaced free of charge within 90 days.

3. Process
To initiate a return, visit our Returns Portal and enter your order ID.
Once approved, a prepaid shipping label will be emailed within 24 hours.
Refunds are processed within 5-7 business days of receiving the item.

4. Exceptions
Perishables, hazardous materials, and customized items cannot be returned.
Sale items are final ─ no returns or exchanges.
Gift cards are non-refundable.
""",

    # MEDIUM ─ structured with markdown tables
    "pricing_table": """
PRICING PLANS ─ CloudBase SaaS Platform

Overview
CloudBase offers three pricing tiers designed for individuals, teams, and enterprises.
All plans include 24/7 uptime monitoring and SSL certificates.

Plan Comparison Table

| Feature                  | Basic      | Pro         | Enterprise    |
|--------------------------|------------|-------------|---------------|
| Price (monthly)          | $9/month   | $29/month   | $99/month     |
| Users                    | 1          | Up to 10    | Unlimited     |
| Storage                  | 5 GB       | 50 GB       | 1 TB          |
| API Calls/day            | 1,000      | 10,000      | Unlimited     |
| Custom Domain            | No         | Yes         | Yes           |
| Priority Support         | No         | Email only  | 24/7 Phone    |
| Analytics Dashboard      | Basic      | Advanced    | Full + Export |
| SLA Uptime Guarantee     | 99.5%      | 99.9%       | 99.99%        |
| SSO / SAML               | No         | No          | Yes           |
| Dedicated Account Manager| No         | No          | Yes           |

Add-On Options
- Extra Storage: $2/GB/month
- Additional API calls: $0.001 per call
- White-labeling: $49/month (Pro and above)

Annual Billing Discount
All plans receive 20% off when billed annually.
Enterprise plans billed annually qualify for a dedicated onboarding session.

Cancellation Policy
Plans can be cancelled anytime. No cancellation fees.
Refunds are prorated for annual plans cancelled within the first 30 days.
""",

    # COMPLEX ─ long, dense, multi-section, tables + technical content
    "research_report": """
TECHNICAL RESEARCH REPORT
Title: Advances in Transformer Architectures for Natural Language Processing

Abstract
This report surveys recent developments in transformer-based architectures,
with a focus on attention mechanisms, positional encodings, and efficiency
improvements. We review key papers and benchmark results across NLP tasks.

1. Attention Mechanisms

1.1 Scaled Dot-Product Attention
The core operation in transformers is scaled dot-product attention:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where Q, K, V are query, key, and value matrices and d_k is the key dimension.
The scaling factor sqrt(d_k) prevents vanishing gradients for large dimensions.

1.2 Multi-Head Attention
Multi-head attention runs h parallel attention functions on projected subspaces:
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) * W_O

This allows the model to jointly attend to information from different subspaces.

1.3 Computational Complexity
Standard self-attention has O(n^2 * d) time complexity where n is sequence length.
This quadratic scaling is a key bottleneck for long documents.

Efficiency Table

| Model Variant       | Attention Type     | Complexity   | Best For          |
|---------------------|--------------------|--------------|-------------------|
| Vanilla Transformer | Full Self-Attention | O(n^2 * d)  | Short sequences   |
| Longformer          | Sliding Window     | O(n * w)     | Long documents    |
| BigBird             | Sparse Attention   | O(n)         | Very long seqs    |
| Linformer           | Low-Rank Approx    | O(n)         | Classification    |
| FlashAttention      | IO-Aware Exact     | O(n^2)       | GPU-optimized     |

2. Positional Encodings

2.1 Absolute Positional Encoding
Original transformers use fixed sinusoidal encodings or learned embeddings.
These do not generalize well to sequences longer than training length.

2.2 Relative Positional Encoding (RoPE)
Rotary Position Embedding (RoPE) encodes position as rotation in embedding space.
Used in LLaMA, GPT-NeoX, and most modern LLMs. Generalizes better to longer contexts.

2.3 ALiBi
Attention with Linear Biases (ALiBi) adds a linear penalty to attention scores.
No position embeddings needed ─ biases are added at inference time.

3. Key Benchmark Results

| Model       | GLUE Score | SQuAD F1 | Parameters | Training FLOPs  |
|-------------|------------|----------|------------|-----------------|
| BERT-base   | 79.6       | 88.5     | 110M       | 1.0x (baseline) |
| RoBERTa     | 88.5       | 94.6     | 125M       | 4.0x            |
| DeBERTa-v3  | 91.9       | 96.0     | 86M        | 2.5x            |
| ELECTRA     | 88.8       | 92.9     | 110M       | 1.0x            |
| T5-base     | 87.1       | 93.1     | 250M       | 8.0x            |

4. Key Findings and Conclusions
- Sparse attention variants reduce complexity but trade off some quality.
- RoPE has become the de-facto positional encoding in decoder-only LLMs.
- FlashAttention achieves full attention quality with better GPU memory efficiency.
- DeBERTa achieves the best GLUE score with fewer parameters than RoBERTa.
- Future research should focus on sub-quadratic exact attention methods.
"""
}


# ── Step 4b: Chunker ────────────────────────────────────────────────

class DocumentChunker:
    def __init__(self, chunk_size=300, overlap=50):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def chunk(self, text: str, doc_name: str) -> list:
        words  = text.split()
        chunks = []
        step   = self.chunk_size - self.overlap
        i = 0
        chunk_id = 0

        while i < len(words):
            chunk_words = words[i: i + self.chunk_size]
            chunks.append({
                "chunk_id"  : f"{doc_name}_chunk_{chunk_id}",
                "doc_name"  : doc_name,
                "text"      : " ".join(chunk_words),
                "word_count": len(chunk_words)
            })
            i += step
            chunk_id += 1

        return chunks

    def chunk_all_docs(self, docs: dict) -> list:
        all_chunks = []
        for name, content in docs.items():
            chunks = self.chunk(content, name)
            all_chunks.extend(chunks)
            print(f"   {name:<20} → {len(chunks)} chunks")
        return all_chunks


# ── Step 4c: Embeddings + FAISS Index ───────────────────────────────

class RAGRetriever:
    def __init__(self, chunks: list, embed_model_name: str, top_k: int = 3):
        self.chunks     = chunks
        self.top_k      = top_k
        self.texts      = [c["text"] for c in chunks]

        print(f"⏳ Loading embedding model: {embed_model_name}")
        self.embedder   = SentenceTransformer(embed_model_name)

        print("🔢 Encoding chunks...")
        embeddings      = self.embedder.encode(
            self.texts, show_progress_bar=True, convert_to_numpy=True
        )
        self.dim        = embeddings.shape[1]

        # L2-normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index      = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        print(f"✅ FAISS index built — {self.index.ntotal} vectors, dim={self.dim}")

    def retrieve(self, query: str) -> list:
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, self.top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    **self.chunks[idx],
                    "similarity_score": round(float(score), 4)
                })
        return results

    def get_context_string(self, query: str) -> str:
        retrieved = self.retrieve(query)
        context   = "\n\n---\n\n".join(
            [f"[Source: {r['doc_name']}]\n{r['text']}" for r in retrieved]
        )
        return context, retrieved


def build_retriever() -> RAGRetriever:
    """Build and return a RAGRetriever from the default DOCS."""
    chunker    = DocumentChunker(CFG["chunk_size"], CFG["chunk_overlap"])
    print("📄 Chunking documents...")
    all_chunks = chunker.chunk_all_docs(DOCS)
    print(f"✅ Total chunks: {len(all_chunks)}")
    retriever  = RAGRetriever(all_chunks, CFG["embed_model"], CFG["top_k_retrieval"])
    return retriever