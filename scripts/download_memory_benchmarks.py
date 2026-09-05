"""Download original public datasets; existing files are left in place."""
import argparse
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "locomo": ("datasets/LoCoMo/locomo10.json",
               "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"),
    "longmemeval": ("datasets/LongMemEval/longmemeval_s_cleaned.json",
                   "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    for dataset in args.datasets:
        relative, url = SOURCES[dataset]
        target = ROOT / relative
        if target.exists():
            print(f"Already present: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".download")
        print(f"Downloading {dataset}...", flush=True)
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        with temporary.open(encoding="utf-8") as handle:
            records = json.load(handle)
        expected = "conversation" if dataset == "locomo" else "haystack_sessions"
        if not isinstance(records, list) or not records or expected not in records[0]:
            raise ValueError(f"Unexpected {dataset} data format")
        temporary.replace(target)
        print(f"Saved {len(records)} histories to {target}")


if __name__ == "__main__":
    main()

