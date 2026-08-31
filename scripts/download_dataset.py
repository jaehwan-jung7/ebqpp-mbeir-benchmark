"""Download real (full-size) EBQPP-MBEIR record files.

These are the actual reconstructed EBQPP records - not the tiny bundled
sample in `examples/`. Each file is a `.pkl` you can load directly with
`src.dataset.EBQPPDataset`.

Currently hosted (top_k=10, M-BEIR "test" split, union candidate pool):
  retriever=uniir,   task=mscoco_task3      (~25 MB,  1000 queries, dim=768)
  retriever=mmembed, task=mscoco_task3      (~98 MB,  1000 queries, dim=4096)
  retriever=uniir,   task=fashion200k_task0 (~42 MB,  dim=768)
  retriever=mmembed, task=fashion200k_task0 (~168 MB, dim=4096)

More tasks can be requested from the authors, or built yourself from raw
UniIR/MM-embed outputs with `scripts/build_dataset.py`.

Usage:
    python scripts/download_dataset.py --retriever uniir --task mscoco_task3
    python scripts/download_dataset.py --retriever mmembed --task fashion200k_task0 --out_dir data/
"""
import argparse
import os
import urllib.request

RELEASE_BASE = "https://github.com/jaehwan-jung7/ebqpp-mbeir-benchmark/releases/download/data-v1"

AVAILABLE = {
    ("uniir", "mscoco_task3"): "ebqpp_uniir_mscoco_task3_test.pkl",
    ("mmembed", "mscoco_task3"): "ebqpp_mmembed_mscoco_task3_test.pkl",
    ("uniir", "fashion200k_task0"): "ebqpp_uniir_fashion200k_task0_test.pkl",
    ("mmembed", "fashion200k_task0"): "ebqpp_mmembed_fashion200k_task0_test.pkl",
}


def download(retriever: str, task: str, out_dir: str):
    key = (retriever, task)
    if key not in AVAILABLE:
        options = "\n".join(f"  --retriever {r} --task {t}" for r, t in AVAILABLE)
        raise SystemExit(f"No hosted file for retriever={retriever}, task={task}. Available:\n{options}")

    filename = AVAILABLE[key]
    url = f"{RELEASE_BASE}/{filename}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    print(f"Downloading {url}\n  -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    print(f"Done ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", type=str, required=True, choices=["uniir", "mmembed"])
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()
    download(args.retriever, args.task, args.out_dir)
