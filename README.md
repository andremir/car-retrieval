# Controlling Authority Retrieval (CAR)

**Paper:** *Controlling Authority Retrieval: A Missing Retrieval Objective for Authority-Governed Knowledge*

Standard retrieval optimizes `argmax s(q, d)` — finding the most relevant document. In authority-governed knowledge bases (legal precedent, security advisories, drug regulations, financial compliance), relevance is structurally wrong: the controlling document is the active frontier of the *authority closure* of the semantic anchor set, not the highest-scoring document. This repository contains the benchmark, scorer, and reference implementations.

---

## The CAR Problem

Given a query `q`, let `A_k(q)` be the top-k semantically similar documents. Define:

- **Authority closure** `cl(A)` = A ∪ {d′ ∈ K : ∃ d ∈ A, d ⊳ d′}  (extend by supersession edges)
- **Active frontier** `front(S)` = {d ∈ S : ∄ d′ ∈ S s.t. d ⊳ d′}  (non-superseded elements)
- **CAR target** `C*_k(q)` = `front(cl(A_k(q)))` — not `argmax s(q, d)`

---

## Datasets

| Domain | Pairs | Corpus | Gold |
|---|---|---|---|
| **FinSuperQA** (SEC compliance) | 1,000 | 12,250 events | Controlling compliance event |
| **GHSA** (security advisories) | 159 | 2,318 docs + distractors | Patch release note |
| **SCOTUS** (legal precedent) | 122 | 244 opinions | Overruling opinion |
| **FDA** (drug labeling) | 500 | 1,000 docs | Superseding recall/label |

All datasets are in `data/`. GHSA, SCOTUS, and FDA are fully public; FinSuperQA is synthetically generated from public SEC filings.

---

## Scorer

```bash
# Reproduce the paper's reference result (TCA@5 = 0.978 on all 1000 examples)
python evaluate.py --retriever two_stage

# Score on the held-out test split (100 examples, stratified by hop type)
python evaluate.py --retriever two_stage --split test --k 5

# Score BM25
python evaluate.py --retriever bm25

# Score dense retriever
python evaluate.py --retriever dense:intfloat/e5-large-v2

# Score custom retriever
python evaluate.py --retriever_path my_retriever.py --output results.json
```

Splits are 80/10/10 stratified by hop type (T0–T3): `train` = 800, `dev` = 100, `test` = 100. Split ID files are in `data/finsuperqa_*_ids.txt`.

**Custom retriever interface** (`my_retriever.py`):
```python
class Retriever:
    name = "my_retriever"
    def __init__(self, corpus: list[dict]): ...
    def retrieve(self, query: str, corpus: list[dict], k: int) -> list[str]:
        # Return ranked list of doc_ids (best first)
        ...
```

**Output format** (stdout or `--output` file):
```json
{
  "retriever": "temporal_two_stage",
  "n": 1000,
  "k": 5,
  "tca_at_k": 0.978,
  "mean_rank": 197.4,
  "mrr": 0.8622
}
```

**Metric:** TCA@k (Temporal Compliance Accuracy) — fraction of examples where the controlling authority document appears in the top-k retrieved set.

---

## Domain Adapters

Each real-world benchmark has a domain adapter in `realworld/`:

| File | Domain | Adapter |
|---|---|---|
| `ghsa_twostage_eval.py` | GHSA CVE pairs | CVE-ID + product entity index |
| `legal_lii_benchmark.py` | SCOTUS precedent | Case citation → overruling opinion |
| `fda_recall_eval.py` | FDA drug labels | NDC/drug-name entity index |
| `llm_downstream_eval.py` | GPT-4o-mini downstream | Dense vs TwoStage answer quality |

All adapters share the same three-attribute interface: `retrieve(query, corpus, k)`, entity index keyed by scope tuple, Stage-1 cutoff `k1`.

---

## Reference Results (TCA@5)

| Retriever | FinSuperQA | GHSA | SCOTUS | FDA |
|---|---|---|---|---|
| BM25 | 0.305 | 0.138 | 0.893 | 0.004 |
| Dense (MiniLM) | 0.214 | 0.270 | 0.172 | 0.064 |
| **TwoStage (BM25)** | **0.978** | **0.975** | **0.926** | **0.774** |
| Entity probe (oracle NER) | 1.000 | 1.000 | — | — |

TwoStage achieves TCA@5 = 0.978 on FinSuperQA (2.9% Stage-1 miss rate); entity probe reaches 1.000 because it uses oracle NER extraction. Dense achieves TCA = 0.000 on T1 (supersession queries) despite Recall@5 > 0.

---

## Downstream LLM Impact (GHSA, n=159)

| Condition | Says Patched | Confident Wrong |
|---|---|---|
| Dense top-5 → GPT-4o-mini | 61% | **39%** |
| TwoStage top-5 → GPT-4o-mini | 82% | 16% |

---

## Install

```bash
pip install -r requirements.txt
```

| Package | Required for |
|---|---|
| `numpy`, `rank-bm25`, `sentence-transformers` | Scorer + FinSuperQA benchmark |
| `requests`, `beautifulsoup4` | Real-world eval scripts (`realworld/`) |
| `openai` | LLM downstream eval (`realworld/llm_downstream_eval.py`) |
| `datasets` | OOD fine-tuning experiment (`falsification/exp5_ood_finetuning.py`) |

---

## Citation

```bibtex
@article{bacellar2026car,
  title   = {Controlling Authority Retrieval: A Missing Retrieval Objective for Authority-Governed Knowledge},
  author  = {Bacellar, Andre},
  year    = {2026}
}
```
