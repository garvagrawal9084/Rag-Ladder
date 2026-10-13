# 🪜 Rag-Ladder

Hands-on RAG experiments in a single notebook. Each rung of the ladder adds one retrieval technique, and every stage runs on the same questions over the same corpus, so you can see what each upgrade actually fixes.

```
Naive semantic search  →  Hybrid (BM25 + RRF)  →  Contextual retrieval  →  Cross-encoder reranking
```

## What's inside

The notebook (`main.ipynb`) builds a RAG pipeline in stages:

| Stage | Technique | What it adds |
|-------|-----------|--------------|
| 1 | **Chunking** (fixed-size vs. recursive) | Recursive chunking keeps section boundaries intact instead of cutting mid-sentence |
| 2 | **Naive RAG** | Embeddings in ChromaDB + cosine similarity, answer generated from the top-k chunks |
| 3 | **Hybrid search** | BM25 keyword search fused with semantic search via Reciprocal Rank Fusion (RRF) |
| 4 | **Contextual retrieval** | An LLM prepends a short context blurb to each chunk before embedding (Anthropic's technique) |
| 5 | **Reranking** | A cross-encoder re-scores the top candidates by reading query and chunk together |
| 6 | **Side-by-side comparison** | The same question run through all three pipelines |

## Why each step exists

- **Naive semantic search can pick the wrong period.** For *"What was ACME's total revenue in Q3 2024?"*, the Q2 report chunk ranked first (distance 0.200) just ahead of the correct Q3 chunk (0.208). Embeddings capture meaning but blur exact tokens like "Q2" vs. "Q3".
- **Hybrid search** brings exact-term matching back (dates, IDs, command names like `/deploy rollback`) while keeping semantic recall.
- **Contextual retrieval** makes a chunk self-describing, e.g. *"This chunk is from the Revenue and Growth section of the ACME Q3 2024 Report..."*, so isolated chunks keep their document and time period.
- **Reranking** is the precision pass. A cross-encoder is slower than a bi-encoder, so it only runs on the shortlist.

## The corpus

17 synthetic ACME Corporation documents (about 90 chunks):

- **5 core documents:** Q3 2024 report, remote work policy, authentication system wiki, deployment process wiki, board AI strategy notes
- **12 distractors:** similar-looking handbook pages, engineering wikis, a Q2 report, a product spec and more, to make retrieval genuinely hard

## Stack

- **LLM and embeddings:** [OpenRouter](https://openrouter.ai) through the OpenAI SDK (`deepseek/deepseek-chat-v3.1` for most generation, `text-embedding-3-small` for embeddings; the final reranked pipeline's generation step uses `gpt-4o-mini`)
- **Vector store:** [ChromaDB](https://www.trychroma.com) (in-memory, cosine space)
- **Keyword search:** [`rank-bm25`](https://github.com/dorianbrown/rank_bm25) (BM25Okapi)
- **Reranker:** [`sentence-transformers`](https://www.sbert.net) CrossEncoder, `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Getting started

```bash
git clone https://github.com/garvagrawal9084/Rag-Ladder.git
cd Rag-Ladder

python -m venv .venv
source .venv/bin/activate

pip install openai chromadb rank-bm25 sentence-transformers python-dotenv numpy jupyter
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
```

Then open the notebook and run the cells top to bottom:

```bash
jupyter notebook main.ipynb
```

> ⚠️ Never commit your `.env` file. Make sure it's listed in `.gitignore`.

## Example: the final comparison

The last cell runs one two-part question through all three pipelines:

```python
Q = "What is the Slack rollback command and what are the security requirements for remote workers?"

native_rag(Q)    # 🔴 semantic only
hybrid_rag(Q)    # 🟡 semantic + BM25 with RRF
full_rag(Q)      # 🟢 contextual chunks + hybrid + reranker
```

The question needs facts from two different documents (the deployment wiki and the remote work policy), which makes it a good stress test for retrieval.

## Notes and caveats

- **Contextualization costs one LLM call per chunk.** For 90 chunks it took a few minutes in the notebook. It's a one-time indexing cost, but it scales with corpus size.
- **The corpus is small and synthetic.** The notebook shows qualitative differences between approaches. It is not a rigorous benchmark.
- **Everything runs in memory.** ChromaDB is created with `chromadb.Client()`, so the index is rebuilt each time the notebook runs.

## Ideas for next steps

- Add a small labeled eval set (question → expected source chunk) and report hit rate / MRR per stage
- Persist the ChromaDB index to disk
- Try a hosted reranker (Cohere, Voyage) against the local cross-encoder
- Add query rewriting or multi-query expansion as another rung

## References

- [Anthropic: Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Reciprocal Rank Fusion (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

## License

MIT (add a `LICENSE` file if you want to use this)
