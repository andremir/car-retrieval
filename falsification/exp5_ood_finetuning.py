"""
Experiment 5: Out-of-Distribution Fine-Tuning (Theorem 2 Gap)
==============================================================

Tests whether a fine-tuned bi-encoder defeats Theorem 2 out-of-distribution.

Hypothesis:
  - Fine-tuning on in-domain (query, superseding_doc) pairs improves in-domain TCA by
    learning implicit entity-scope conditioning — i.e., the encoder learns that
    "patch text" should be close to "query about that CVE" because they share entity scope.
  - Out-of-distribution (different vocabulary regime), this learned signal is absent:
    FDA recall notices use administrative vocabulary that GHSA-trained models don't recognize.
  - Therefore: fine-tuned TCA stays high in-domain but degrades OOD → both approaches
    collapse to (1/κ)·R_anchor in the OOD adversarial regime → Theorem 2 holds for the
    full instance class, even when fine-tuning helps specific subsets.

Setup:
  - Train:  80% GHSA pairs (127 pairs), fine-tune MiniLM
  - In-domain test:  20% GHSA pairs (32 pairs)
  - OOD test:  FDA recall pairs (500 pairs), different vocabulary regime

If fine-tuned OOD TCA ≈ zero-shot OOD TCA << Two-Stage:
  → Fine-tuning improves by learning entity scope signals, not by defeating CAR structure.
  → Supports Theorem 2's claim about the *instance class*, not just untrained models.

If fine-tuned OOD TCA ≈ Two-Stage:
  → Fine-tuning generalizes (entity scope signals are domain-universal).
  → Theorem 2 needs caveat; paper should acknowledge.

Dependencies: pip install sentence-transformers[train] rank-bm25 numpy
Artifacts: data/falsification_ood_finetuning.json
Runtime: ~15 min (fine-tuning ~3 epochs on 127 pairs)
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np

ROOT        = Path(__file__).parent.parent
PAIRS_PATH  = ROOT / "data" / "ghsa_real_pairs.json"
FREEFORM    = ROOT / "data" / "ghsa_freeform_cache.jsonl"
FDA_PATH    = ROOT / "data" / "fda_recall_pairs.json"
MODEL_DIR   = ROOT / "data" / "finetuned_minilm_ghsa"
OUT_PATH    = ROOT / "data" / "falsification_ood_finetuning.json"

TRAIN_FRAC  = 0.80
SEED        = 42
EPOCHS      = 3
BATCH_SIZE  = 16
K           = 5


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------
def build_ghsa_corpus(pairs, n_distractors=2000, seed=42):
    docs, metas = [], []
    for p in pairs:
        docs.append(p["disclosure_text"])
        metas.append({"type": "disclosure", "product": p["product"], "cve_id": p["cve_id"]})
    patch_start = len(docs)
    for i, p in enumerate(pairs):
        docs.append(p["patch_text"])
        metas.append({"type": "patch", "product": p["product"], "cve_id": p["cve_id"]})
    rng = random.Random(seed)
    products = ["nginx","openssl","log4j","redis","django","flask","rails","postgresql","curl"]
    attacks  = ["remote code execution","SQL injection","buffer overflow"]
    sevs     = ["Critical","High","Medium"]
    for i in range(n_distractors):
        prod = rng.choice(products); cve = f"CVE-2023-{90000+i:05d}"; ver = f"{rng.randint(1,9)}.{rng.randint(0,9)}"
        docs.append(f"{cve} in {prod}. {rng.choice(attacks)}. {rng.choice(sevs)}. Versions ≤{ver}. No patch.")
        metas.append({"type": "distractor", "product": prod, "cve_id": cve})
    return docs, metas, patch_start


def build_fda_corpus(pairs, n_distractors=2000, seed=99):
    docs = [p["anchor_text"] for p in pairs]   # approved drug labels
    patch_start = len(docs)
    docs.extend(p["superseder_text"] for p in pairs)  # recall notices
    rng = random.Random(seed)
    drug_classes = ["antihypertensive","antibiotic","antidiabetic","analgesic","anticoagulant"]
    for i in range(n_distractors):
        cls = rng.choice(drug_classes)
        docs.append(f"Drug class: {cls}. Lot number {rng.randint(10000,99999)}. "
                    f"Manufactured by Generic Pharma Co. Expiry 2026. No recall action required.")
    return docs, patch_start


# ---------------------------------------------------------------------------
# Evaluate TCA@K for a sentence-transformer model
# ---------------------------------------------------------------------------
def eval_tca(model, pairs, queries: dict[str, str], docs: list[str],
             pair_indices: list[int], patch_start: int) -> float:
    doc_embs = model.encode(docs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    hits = []
    for pi in pair_indices:
        p = pairs[pi]
        pid = p["id"]
        q = queries.get(pid)
        if q is None:
            continue
        gold_idx = patch_start + pi
        q_emb = model.encode([q], normalize_embeddings=True)
        scores = (q_emb @ doc_embs.T)[0]
        top5 = list(np.argsort(-scores)[:K])
        hits.append(int(gold_idx in top5))
    return sum(hits) / len(hits) if hits else 0.0


def eval_tca_fda(model, fda_pairs: list[dict], docs: list[str], patch_start: int) -> float:
    doc_embs = model.encode(docs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    hits = []
    for i, p in enumerate(fda_pairs):
        q = p["query"]
        gold_idx = patch_start + i
        q_emb = model.encode([q], normalize_embeddings=True)
        scores = (q_emb @ doc_embs.T)[0]
        top5 = list(np.argsort(-scores)[:K])
        hits.append(int(gold_idx in top5))
    return sum(hits) / len(hits) if hits else 0.0


# ---------------------------------------------------------------------------
# Two-Stage TCA (BM25 + entity-indexed) — reference ceiling
# ---------------------------------------------------------------------------
def eval_twostage_ghsa(pairs, queries, docs, metas, pair_indices, patch_start):
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    entity_index = {}
    for i in range(patch_start, patch_start + len(pairs)):
        m = metas[i]
        entity_index[(m["product"].lower(), m["cve_id"].upper())] = i
    hits = []
    for pi in pair_indices:
        p = pairs[pi]
        pid = p["id"]
        q = queries.get(pid)
        if q is None:
            continue
        gold_idx = patch_start + pi
        bm25_scores = bm25.get_scores(tokenize(q))
        top_k1 = list(np.argsort(-bm25_scores)[:5])
        candidates = []
        seen = set()
        for idx in top_k1:
            m = metas[idx]
            if m["type"] == "disclosure":
                key = (m["product"].lower(), m["cve_id"].upper())
                patch_idx = entity_index.get(key)
                if patch_idx is not None and patch_idx not in seen:
                    candidates.append(patch_idx); seen.add(patch_idx)
        hits.append(int(gold_idx in (candidates + [d for d in top_k1 if d not in seen])[:K]))
    return sum(hits) / len(hits) if hits else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from sentence_transformers import SentenceTransformer, losses

    pairs_all = json.load(open(PAIRS_PATH))
    freeform: dict[str, str] = {}
    for line in open(FREEFORM):
        if line.strip():
            e = json.loads(line); freeform[e["id"]] = e["query"]
    fda_pairs = json.load(open(FDA_PATH))
    print(f"Loaded {len(pairs_all)} GHSA pairs, {len(freeform)} queries, {len(fda_pairs)} FDA pairs")

    # 80/20 split by pair index
    rng = random.Random(SEED)
    indices = list(range(len(pairs_all)))
    rng.shuffle(indices)
    n_train = int(len(indices) * TRAIN_FRAC)
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]
    print(f"Train: {len(train_idx)} pairs  Test: {len(test_idx)} pairs")

    # Build GHSA corpus (full pairs used for corpus, train_idx for training)
    ghsa_docs, ghsa_metas, ghsa_patch_start = build_ghsa_corpus(pairs_all)

    # Zero-shot baseline
    print("\n[Zero-shot baseline]")
    zs_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    zs_ghsa_train = eval_tca(zs_model, pairs_all, freeform, ghsa_docs, train_idx, ghsa_patch_start)
    zs_ghsa_test  = eval_tca(zs_model, pairs_all, freeform, ghsa_docs, test_idx,  ghsa_patch_start)
    fda_docs, fda_patch_start = build_fda_corpus(fda_pairs)
    zs_fda = eval_tca_fda(zs_model, fda_pairs, fda_docs, fda_patch_start)
    print(f"  Zero-shot MiniLM  GHSA-train={zs_ghsa_train:.3f}  GHSA-test={zs_ghsa_test:.3f}  FDA-OOD={zs_fda:.3f}")

    # Two-Stage ceiling on test set
    ts_test = eval_twostage_ghsa(pairs_all, freeform, ghsa_docs, ghsa_metas, test_idx, ghsa_patch_start)
    print(f"  Two-Stage ceiling GHSA-test={ts_test:.3f}  (reference)")

    # Fine-tuning
    print("\n[Fine-tuning MiniLM on GHSA train split…]")
    from datasets import Dataset as HFDataset
    train_data: dict[str, list] = {"anchor": [], "positive": [], "negative": []}
    for pi in train_idx:
        p = pairs_all[pi]
        q = freeform.get(p["id"])
        if q is None:
            continue
        train_data["anchor"].append(q)
        train_data["positive"].append(p["patch_text"])
        # Hard negative: disclosure text (wrong type, same entity)
        train_data["negative"].append(p["disclosure_text"])

    train_dataset = HFDataset.from_dict(train_data)
    print(f"  Training examples: {len(train_dataset)}")

    ft_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    loss_fn  = losses.MultipleNegativesRankingLoss(ft_model)

    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    args = SentenceTransformerTrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        save_strategy="no",
        logging_steps=50,
    )
    trainer = SentenceTransformerTrainer(
        model=ft_model,
        args=args,
        train_dataset=train_dataset,
        loss=loss_fn,
    )
    trainer.train()
    print("  Fine-tuning complete.")

    # Evaluate fine-tuned model
    print("\n[Fine-tuned model evaluation]")
    ft_ghsa_train = eval_tca(ft_model, pairs_all, freeform, ghsa_docs, train_idx, ghsa_patch_start)
    ft_ghsa_test  = eval_tca(ft_model, pairs_all, freeform, ghsa_docs, test_idx,  ghsa_patch_start)
    ft_fda        = eval_tca_fda(ft_model, fda_pairs, fda_docs, fda_patch_start)
    print(f"  Fine-tuned MiniLM GHSA-train={ft_ghsa_train:.3f}  GHSA-test={ft_ghsa_test:.3f}  FDA-OOD={ft_fda:.3f}")

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    delta_test = ft_ghsa_test - zs_ghsa_test
    delta_fda  = ft_fda - zs_fda
    print(f"  In-domain improvement (GHSA test): {delta_test:+.3f}  ({'+' if delta_test>0 else ''}{delta_test*100:.1f}pp)")
    print(f"  OOD change (FDA):                  {delta_fda:+.3f}  ({'+' if delta_fda>0 else ''}{delta_fda*100:.1f}pp)")
    print(f"  Two-Stage ceiling (GHSA test):     {ts_test:.3f}")
    print(f"  Fine-tuned gap to Two-Stage:       {ts_test - ft_ghsa_test:.3f}")
    print()

    if delta_test > 0.05 and abs(delta_fda) < 0.05:
        print("[PASS] Fine-tuning improves in-domain (+{:.0f}pp) but does not generalize OOD ({:+.0f}pp).".format(
            delta_test*100, delta_fda*100))
        print("       Supports Theorem 2: fine-tuning learns implicit scope conditioning, domain-specific.")
    elif delta_fda > 0.15:
        print("[FAIL] Fine-tuning generalizes OOD by {:.0f}pp. Entity-scope signals may be domain-universal.".format(delta_fda*100))
        print("       Add caveat to Theorem 2; investigate what vocabulary transfers.")
    else:
        print("[PARTIAL] Mixed result. Inspect per-query breakdown.")
    print("="*70)

    out = {
        "train_n": len(train_idx), "test_n": len(test_idx), "fda_n": len(fda_pairs),
        "epochs": EPOCHS,
        "zero_shot": {
            "ghsa_train_tca5": round(zs_ghsa_train, 4),
            "ghsa_test_tca5": round(zs_ghsa_test, 4),
            "fda_ood_tca5": round(zs_fda, 4),
        },
        "fine_tuned": {
            "ghsa_train_tca5": round(ft_ghsa_train, 4),
            "ghsa_test_tca5": round(ft_ghsa_test, 4),
            "fda_ood_tca5": round(ft_fda, 4),
        },
        "two_stage_ceiling": {
            "ghsa_test_tca5": round(ts_test, 4),
        },
        "delta": {
            "in_domain_pp": round(delta_test * 100, 1),
            "ood_pp": round(delta_fda * 100, 1),
            "gap_to_twostage_pp": round((ts_test - ft_ghsa_test) * 100, 1),
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
