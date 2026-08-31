"""Quickstart: load a bundled EBQPP-MBEIR sample and inspect it.

No download needed - this uses the small pre-built sample checked into
`examples/sample_data/`. It also runs a toy unsupervised QPP baseline
(top-1 retriever score) and reports its correlation with the ground-truth
nDCG@10 label, just to show what the benchmark is used for.

Run:
    python examples/load_example.py
"""
import os
import sys

import numpy as np
from scipy.stats import kendalltau, pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import DATASET_ID_TO_NAME, MODALITY_ID_TO_NAME, EBQPPDataset

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "ebqpp_sample_uniir_mscoco_task3.pkl")


def main():
    ds = EBQPPDataset(SAMPLE_PATH)
    print(f"Loaded {len(ds)} EBQPP records from {os.path.basename(SAMPLE_PATH)}")

    r0 = ds[0]
    print("\nExample record:")
    print(f"  dataset      = {DATASET_ID_TO_NAME[r0['dataset_id']]} (task_id={r0['task_id']})")
    print(f"  query id     = {r0['qid_unhash']} (modality={MODALITY_ID_TO_NAME[r0['q_modality']]})")
    print(f"  top-{len(r0['did_unhash'])} candidates = {r0['did_unhash']}")
    print(f"  relevance    = {r0['rg'].tolist()}")
    print(f"  embed shape  = {tuple(r0['embed'].shape)}  (row 0 = query, rows 1.. = candidates)")
    print(f"  nDCG@10      = {r0['ndcg10'].item():.4f}")

    # Toy unsupervised QPP baseline: use the top-1 retriever score to predict
    # query performance, and check how well it correlates with actual nDCG@10.
    top1_scores = np.array([r["score"][0].item() for r in ds])
    actual_ndcg10 = np.array([r["ndcg10"].item() for r in ds])

    pearson_r, _ = pearsonr(top1_scores, actual_ndcg10)
    kendall_tau, _ = kendalltau(top1_scores, actual_ndcg10)
    print(f"\nBaseline QPP method: top-1 retriever score")
    print(f"  Pearson r   with actual nDCG@10: {pearson_r:.4f}")
    print(f"  Kendall tau with actual nDCG@10: {kendall_tau:.4f}")
    print("\nThis is the EBQPP task: given only the `embed`/`score` fields, predict")
    print("`ndcg10` (or hit/rr/etc.) without access to the ground-truth `rg` labels.")


if __name__ == "__main__":
    main()
