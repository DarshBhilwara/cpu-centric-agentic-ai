#!/usr/bin/env python3
"""
All datasets are returned in a unified format:

{
    "id": str,
    "question": str,
    "gold_answers": list,
    "choices": list | None,
    "metadata": dict
}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Optional


DATASET_DIR = Path(__file__).parent / "datasets"

DATASETS = {
    "freshqa": DATASET_DIR / "freshqa.jsonl",
    "qasc": DATASET_DIR / "qasc.jsonl",
}


def _read_jsonl(path: Path) -> List[dict]:

    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            data.append(json.loads(line))

    return data


def _parse_freshqa(records: List[dict]) -> List[dict]:
    parsed = []

    for sample in records:

        parsed.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "gold_answers": sample.get("gold_answers", []),
                "choices": None,
                "metadata": sample.get("metadata", {}),
            }
        )

    return parsed


def _parse_qasc(records: List[dict]) -> List[dict]:
    parsed = []

    for sample in records:

        parsed.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "gold_answers": sample.get("gold_answers", []),
                "choices": sample.get("choices", []),
                "metadata": sample.get("metadata", {}),
            }
        )

    return parsed



def load_dataset(benchmark: str, num_queries: Optional[int] = None, shuffle: bool = False, seed: int = 42,) -> List[Dict]:

    benchmark = benchmark.lower()

    if benchmark not in DATASETS:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Supported: {list(DATASETS.keys())}"
        )

    path = DATASETS[benchmark]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    raw = _read_jsonl(path)

    if benchmark == "freshqa":
        dataset = _parse_freshqa(raw)

    elif benchmark == "qasc":
        dataset = _parse_qasc(raw)

    else:
        raise RuntimeError("Unsupported dataset")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(dataset)

    if num_queries is not None:
        dataset = dataset[:num_queries]

    return dataset


def make_batches(dataset: List[Dict], batch_size: int):

    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark",
        choices=["freshqa", "qasc"],
        required=True,
    )

    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
    )

    args = parser.parse_args()

    dataset = load_dataset(
        benchmark=args.benchmark,
        num_queries=args.num_queries,
        shuffle=args.shuffle,
    )

    print("=" * 60)
    print(f"Benchmark : {args.benchmark}")
    print(f"Questions : {len(dataset)}")
    print("=" * 60)

    print()

    print("First Question")
    print("-" * 60)
    print(dataset[0]["question"])

    print()

    print("Batch Sizes")

    for i, batch in enumerate(make_batches(dataset, args.batch_size)):
        print(f"Batch {i+1}: {len(batch)}")