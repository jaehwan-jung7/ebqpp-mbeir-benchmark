"""Download a small, real sample of raw UniIR/MM-embed retrieval outputs
(run file, qrels, query jsonl, query embeddings, and a filtered candidate
pool) for the M-BEIR "mscoco_task3" test task - 300 of its 5000 queries.

This is what `scripts/build_dataset.py` expects as input. It exists so you
can run the full reconstruction pipeline end-to-end on real data without
having your own UniIR/MM-embed outputs first; see the README's "Building
Your Own EBQPP Dataset" section for the exact `build_dataset.py` command.

Usage:
    python scripts/download_raw_sample.py --out_dir data/raw_sample
"""
import argparse
import os
import urllib.request
import zipfile

RELEASE_URL = "https://github.com/jaehwan-jung7/ebqpp-mbeir-benchmark/releases/download/data-v1/ebqpp_raw_sample_mscoco_task3.zip"


def download(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "raw_sample.zip")

    print(f"Downloading {RELEASE_URL}\n  -> {zip_path}")
    urllib.request.urlretrieve(RELEASE_URL, zip_path)

    print(f"Extracting into {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    os.remove(zip_path)

    print(f"Done. Contents of {out_dir}:")
    for name in sorted(os.listdir(out_dir)):
        print(" ", name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default="data/raw_sample")
    args = parser.parse_args()
    download(args.out_dir)
