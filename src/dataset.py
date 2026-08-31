"""Loading utilities for EBQPP-MBEIR records.

Each record is a dict produced by `scripts/build_dataset.py` (see that file
for how records are built from raw UniIR / MM-embed retrieval outputs):

    q_modality   int    query modality id (0=text, 1=image, 2=image+text)
    d_modality   Tensor [k]    modality id of each top-k candidate
    task_id      int    M-BEIR task id (0,1,2,3,4,6,7,8)
    dataset_id   int    M-BEIR dataset id (see DATASET_ID_TO_NAME)
    qid          int    hashed query id
    qid_unhash   str    original "{dataset_id}:{index}" query id
    did          Tensor [k]    hashed candidate ids (top-k)
    did_unhash   list[str]     original candidate ids (top-k)
    rg           Tensor [k]    binary relevance of each top-k candidate
    score        Tensor [100]  retriever similarity scores (top-100)
    hit{5,10,20,50,100}   Tensor[]  binary hit@k
    rr{5,10,20,50,100}    Tensor[]  reciprocal rank@k
    ndcg{5,10,20,50,100}  Tensor[]  nDCG@k (ground-truth QPP label)
    idcg{5,10,20,50,100}  Tensor[]  ideal DCG@k
    embed        Tensor [1+k, D]  float16, row 0 = query embedding,
                                  rows 1..k = candidate embeddings
"""
import torch
from torch.utils.data import Dataset

DATASET_ID_TO_NAME = {
    0: "visualnews", 1: "fashion200k", 2: "webqa", 3: "edis", 4: "nights",
    5: "oven", 6: "infoseek", 7: "fashioniq", 8: "cirr", 9: "mscoco",
}
MODALITY_ID_TO_NAME = {0: "text", 1: "image", 2: "image,text"}


class EBQPPDataset(Dataset):
    """Thin wrapper around a `dataset_*.pkl` file (a list of record dicts)."""

    def __init__(self, pkl_path: str):
        self.records = torch.load(pkl_path, map_location="cpu", weights_only=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_ebqpp(batch):
    """Stack a list of records into batched tensors. Works for any top_k
    since it only stacks fields that are already tensors of matching shape."""
    out = {
        "q_modality": torch.tensor([b["q_modality"] for b in batch], dtype=torch.long),
        "task_id": torch.tensor([b["task_id"] for b in batch], dtype=torch.long),
        "dataset_id": torch.tensor([b["dataset_id"] for b in batch], dtype=torch.long),
        "qid": torch.tensor([b["qid"] for b in batch], dtype=torch.long),
        "d_modality": torch.stack([b["d_modality"] for b in batch], 0),
        "did": torch.stack([b["did"] for b in batch], 0),
        "rg": torch.stack([b["rg"] for b in batch], 0),
        "score": torch.stack([b["score"] for b in batch], 0),
        "embed": torch.stack([b["embed"] for b in batch], 0),
    }
    for k in ("5", "10", "20", "50", "100"):
        for prefix in ("hit", "rr", "ndcg", "idcg"):
            key = f"{prefix}{k}"
            if key in batch[0]:
                out[key] = torch.stack([b[key] for b in batch], 0)
    return out
