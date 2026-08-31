"""Reconstruct an EBQPP-MBEIR record file from raw UniIR / MM-embed retrieval outputs.

This is the pipeline used to build the `.pkl` files served by
`scripts/download_dataset.py`. You only need this script if you want to
build EBQPP records for a *new* M-BEIR task/dataset or a different
underlying retriever, from your own run file + embeddings.

Required raw inputs (produced by UniIR or MM-embed for one M-BEIR task):
  --run_path          TREC-style run file: "qid Q0 did rank score run_id task_id"
  --qrels_path        TREC-style qrels file: "qid Q0 did grade task_id"
  --query_jsonl_path  M-BEIR query jsonl (has "id"/"query_modality" fields)
  --query_embed_path  .npy, query embeddings, shape [Nq, D]
  --query_ids_path    .npy, hashed query ids aligned with query_embed_path
  --cand_embed_path   .npy, candidate-pool embeddings, shape [Nc, D]
  --cand_ids_path     .npy, hashed candidate ids aligned with cand_embed_path
  --cand_meta_path    optional JSON {did_unhash: {"modality": "text"|"image"|...}};
                       if omitted, candidate modality defaults to "text"

Example:
  python scripts/build_dataset.py \\
    --run_path data/raw/mbeir_mscoco_task3_union_pool_test_k100_run.txt \\
    --qrels_path data/raw/mbeir_mscoco_task3_test_qrels.txt \\
    --query_jsonl_path data/raw/mbeir_mscoco_task3_test.jsonl \\
    --query_embed_path data/raw/mbeir_mscoco_task3_test_embed.npy \\
    --query_ids_path data/raw/mbeir_mscoco_task3_test_ids.npy \\
    --cand_embed_path data/raw/mbeir_union_cand_pool_embed.npy \\
    --cand_ids_path data/raw/mbeir_union_cand_pool_ids.npy \\
    --top_k 10 \\
    --out_path data/dataset_mscoco_task3_test.pkl
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

DATASET_CAN_NUM_UPPER_BOUND = 10_000_000
DATASET_QUERY_NUM_UPPER_BOUND = 500_000
MODALITY2ID = {"text": 0, "image": 1, "image,text": 2}


def hash_qid(qid_unhash: str) -> int:
    d, i = map(int, qid_unhash.split(":"))
    return d * DATASET_QUERY_NUM_UPPER_BOUND + i


def hash_did(did_unhash: str) -> int:
    d, i = map(int, did_unhash.split(":"))
    return d * DATASET_CAN_NUM_UPPER_BOUND + i


def modality_to_id(modality):
    modality = (modality or "text").lower().replace(" ", "")
    if modality == "text,image":
        modality = "image,text"
    return MODALITY2ID.get(modality, 0)


def hit_at_k(rgs, k):
    k = min(k, len(rgs))
    return 1.0 if k > 0 and any(g > 0 for g in rgs[:k]) else 0.0


def rr_at_k(rgs, k):
    k = min(k, len(rgs))
    for i in range(k):
        if rgs[i] > 0:
            return 1.0 / (i + 1)
    return 0.0


def _discounts(k):
    return np.array([1.0 / math.log2(i + 2) for i in range(k)], dtype=np.float64)


def dcg_at_k_binary(rgs, k):
    k = min(k, len(rgs))
    if k <= 0:
        return 0.0
    rel = np.asarray(rgs[:k]) > 0
    return float((_discounts(k) * rel.astype(np.float64)).sum())


def idcg_at_k_from_all_true_binary(all_true, k):
    t = int(np.sum(np.asarray(all_true) > 0))
    t = max(0, min(t, k))
    return float(_discounts(t).sum()) if t > 0 else 0.0


def ndcg_at_k_binary(rgs, all_true, k):
    dcg = dcg_at_k_binary(rgs, k)
    idcg = idcg_at_k_from_all_true_binary(all_true, k)
    return dcg / idcg if idcg > 0 else 0.0


def load_qrels_txt(path):
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid_unhash, _zero, did_unhash, grade, _task = line.strip().split()
            qrels[qid_unhash][did_unhash] = int(grade)
    return qrels


def load_query_meta(query_jsonl_path):
    meta = {}
    with open(query_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            qid_unhash = o.get("id") or o.get("qid") or o.get("query_id")
            if qid_unhash is None:
                continue
            meta[qid_unhash] = {"modality": o.get("query_modality")}
    return meta


def build(args):
    qrels = load_qrels_txt(args.qrels_path)
    qmeta = load_query_meta(args.query_jsonl_path)

    q_embeds = np.load(args.query_embed_path).astype(args.embed_dtype)
    q_ids = np.load(args.query_ids_path)
    c_embeds = np.load(args.cand_embed_path).astype(args.embed_dtype)
    c_ids = np.load(args.cand_ids_path).astype(np.int64)

    qid2row = {int(h): r for r, h in enumerate(q_ids)}

    cand_meta = {}
    if args.cand_meta_path:
        with open(args.cand_meta_path, "r", encoding="utf-8") as f:
            cand_meta = json.load(f)  # did_unhash -> {"modality": ...}

    with open(args.run_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    by_qid = defaultdict(list)
    q_task = {}
    top_k = args.top_k
    for l in lines:
        qid_unhash, _, did_unhash, rank, score, _run_id, task_id = l.split()
        by_qid[qid_unhash].append((int(rank), did_unhash, hash_did(did_unhash), float(score)))
        q_task[qid_unhash] = int(task_id)

    need_h_dids = np.fromiter(
        {h for triples in by_qid.values() for r, _, h, _ in triples if r <= top_k},
        dtype=np.int64,
    )
    sort_idx = np.argsort(c_ids)
    sort_ids = c_ids[sort_idx]
    pos = np.searchsorted(sort_ids, need_h_dids)
    assert np.all(sort_ids[pos] == need_h_dids), "some candidate ids missing from candidate pool"
    hdid2row = {int(h): int(sort_idx[p]) for h, p in zip(need_h_dids, pos)}

    records = []
    for qid_unhash, triples in tqdm(by_qid.items(), desc="building records"):
        if qid_unhash not in qrels:
            continue

        triples.sort(key=lambda x: x[0])
        top_docs = triples[:top_k]
        did_unhash_list = [t[1] for t in top_docs]
        did_h_list = [t[2] for t in top_docs]
        scores = [t[3] for t in triples[:100]]

        rels_for_q = qrels[qid_unhash]
        rgs = [rels_for_q.get(d, 0) for d in did_unhash_list]
        all_true = list(rels_for_q.values())

        ks = (5, 10, 20, 50, 100)
        metrics = {}
        for k in ks:
            metrics[f"hit{k}"] = hit_at_k(rgs, k)
            metrics[f"rr{k}"] = rr_at_k(rgs, k)
            metrics[f"ndcg{k}"] = ndcg_at_k_binary(rgs, all_true, k)
            metrics[f"idcg{k}"] = idcg_at_k_from_all_true_binary(all_true, k)

        qid_h = hash_qid(qid_unhash)
        q_row = qid2row[qid_h]
        q_vec = q_embeds[q_row : q_row + 1, :]
        d_mat = c_embeds[[hdid2row[h] for h in did_h_list]]
        embed = torch.from_numpy(np.vstack([q_vec, d_mat]))

        meta = qmeta.get(qid_unhash) or {}
        d_modality = [
            modality_to_id((cand_meta.get(d) or {}).get("modality")) for d in did_unhash_list
        ]
        records.append(
            {
                "q_modality": modality_to_id(meta.get("modality")),
                "d_modality": torch.tensor(d_modality, dtype=torch.long),
                "task_id": q_task.get(qid_unhash, 0),
                "dataset_id": int(qid_unhash.split(":", 1)[0]),
                "qid": int(qid_h),
                "qid_unhash": qid_unhash,
                "did": torch.tensor(did_h_list, dtype=torch.long),
                "did_unhash": did_unhash_list,
                "rg": torch.tensor(rgs, dtype=torch.long),
                "score": torch.tensor(scores, dtype=torch.float32),
                **{k: torch.tensor(v, dtype=torch.float32) for k, v in metrics.items()},
                "embed": embed,
            }
        )

    torch.save(records, args.out_path)
    print(f"saved {len(records)} records -> {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--qrels_path", type=str, required=True)
    parser.add_argument("--query_jsonl_path", type=str, required=True)
    parser.add_argument("--query_embed_path", type=str, required=True)
    parser.add_argument("--query_ids_path", type=str, required=True)
    parser.add_argument("--cand_embed_path", type=str, required=True)
    parser.add_argument("--cand_ids_path", type=str, required=True)
    parser.add_argument("--cand_meta_path", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--embed_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--out_path", type=str, required=True)
    build(parser.parse_args())
