# EBQPP-MBEIR Benchmark

Embedding-Based Query Performance Prediction (EBQPP) benchmark data, built by
reconstructing [M-BEIR](https://github.com/TIGER-AI-Lab/UniIR) (a multimodal
information retrieval benchmark) around two underlying Universal Multimodal
Retrievers (UMR):

- **UniIR** (CLIP-SF, embedding dim 768)
- **MM-embed** (LLaVA-Next-7B backbone, embedding dim 4096)

For each query, we keep its top-k retrieved candidates from the underlying
retriever together with the query/candidate embeddings and the actual
retrieval quality (nDCG, Hit, RR @ 5/10/20/50/100). This is the label an
embedding-based QPP model has to predict from the embeddings alone, without
seeing the ground-truth relevance.

Contents
- [Environment Setup](#environment-setup)
- [Quickstart](#quickstart)
- [Data Schema](#data-schema)
- [Downloading the Full Benchmark Data](#downloading-the-full-benchmark-data)
- [Building Your Own EBQPP Dataset (Advanced)](#building-your-own-ebqpp-dataset-advanced)
- [Acknowledgement](#acknowledgement)

## Environment Setup
```bash
pip install -r requirements.txt
```

## Quickstart
No download required. `examples/sample_data/` bundles 150 real EBQPP records
(task: MSCOCO image-to-text retrieval, retriever: UniIR, top_k=10):
```bash
python examples/load_example.py
```
This loads the sample, prints one record's fields, and runs a toy
unsupervised QPP baseline (top-1 retriever score) to show how its
correlation with the actual nDCG@10 label is measured - the same
protocol used to evaluate a trained EBQPP model.

## Data Schema
Each record (see `src/dataset.py`) is a dict:

| field | type | description |
|---|---|---|
| `q_modality` | int | query modality (0=text, 1=image, 2=image+text) |
| `d_modality` | Tensor `[k]` | modality of each top-k candidate |
| `task_id`, `dataset_id` | int | M-BEIR task / dataset identifiers |
| `qid_unhash`, `did_unhash` | str / list[str] | original M-BEIR ids |
| `rg` | Tensor `[k]` | binary relevance of each top-k candidate |
| `score` | Tensor `[100]` | retriever similarity scores (top-100) |
| `hit{k}`, `rr{k}`, `ndcg{k}`, `idcg{k}` | Tensor | retrieval quality @ k in {5,10,20,50,100} |
| `embed` | Tensor `[1+k, D]`, float16 | row 0 = query embedding, rows 1.. = candidate embeddings |

Load any record file with:
```python
from src.dataset import EBQPPDataset
ds = EBQPPDataset("path/to/dataset_xxx.pkl")
```

## Downloading the Full Benchmark Data
The bundled sample is a 150-record slice. Full per-task record files
(1000 queries per task, top_k=10, M-BEIR "test" split) are hosted as
GitHub Release assets:
```bash
python scripts/download_dataset.py --retriever uniir   --task mscoco_task3
python scripts/download_dataset.py --retriever mmembed --task mscoco_task3
python scripts/download_dataset.py --retriever uniir   --task fashion200k_task0
python scripts/download_dataset.py --retriever mmembed --task fashion200k_task0
```
Files are saved to `data/` by default. More tasks can be built yourself
(see below) from raw UniIR/MM-embed outputs, or requested from the authors.

## Building Your Own EBQPP Dataset (Advanced)
`scripts/build_dataset.py` is the pipeline that turns raw UniIR/MM-embed
retrieval outputs (a TREC-style run file, qrels, and query/candidate
embeddings) into an EBQPP record file. To try it end-to-end on real data
with nothing you need to provide yourself, download a small real sample
(300 of the 5000 `mscoco_task3` test queries, with a candidate pool
filtered down to just what those queries need) and run it directly:
```bash
python scripts/download_raw_sample.py

python scripts/build_dataset.py \
  --run_path data/raw_sample/uniir/mbeir_mscoco_task3_single_pool_test_k10_run.txt \
  --qrels_path data/raw_sample/mbeir_mscoco_task3_test_qrels.txt \
  --query_jsonl_path data/raw_sample/mbeir_mscoco_task3_test.jsonl \
  --query_embed_path data/raw_sample/uniir/mbeir_mscoco_task3_test_embed.npy \
  --query_ids_path data/raw_sample/uniir/mbeir_mscoco_task3_test_ids.npy \
  --cand_embed_path data/raw_sample/uniir/mbeir_mscoco_task3_cand_pool_embed.npy \
  --cand_ids_path data/raw_sample/uniir/mbeir_mscoco_task3_cand_pool_ids.npy \
  --top_k 10 \
  --out_path data/dataset_mscoco_task3_test_uniir.pkl
```
The same sample also has an MM-embed version - swap the `uniir/` paths above
for `mmembed/mbeir_mscoco_task3_union_pool_test_k100_run.txt` and the
matching `mmembed/` embedding files.

This `mscoco_task3` sample is only for seeing the pipeline work on real
data - it's already covered by `scripts/download_dataset.py` above. To
extend the benchmark to a **new** M-BEIR task, run UniIR or MM-embed
yourself on that task to get its own run file/qrels/embeddings, then point
`build_dataset.py` at those files the same way. See the header of
`scripts/build_dataset.py` for the exact input format.

## Acknowledgement
This repository was developed with support from the **서울시립대학교 데이터 사이언스 플러스 차세대 융합인재 양성사업단** - http://dsplus.uos.ac.kr/
