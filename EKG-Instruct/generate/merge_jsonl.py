import json
import os
import sys
from typing import List


def merge_jsonl_files(input_paths: List[str], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as out_f:
        for path in input_paths:
            with open(path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    out_f.write(line if line.endswith("\n") else line + "\n")

def main() -> None:
    base_dir = "/mnt/disk/yh24/test1/graphrag-purity/"
    inputs = [
        os.path.join(base_dir, "merged_sft33.jsonl"),
        os.path.join(base_dir, "output_synthetic_dataset4.jsonl"),
    ]
    output = os.path.join(base_dir, "merged_sft44.jsonl")

    for p in inputs:
        if not os.path.isfile(p):
            print(f"Input file not found: {p}", file=sys.stderr)
            sys.exit(1)

    merge_jsonl_files(inputs, output)
    print(f"Merged {len(inputs)} files into: {output}")


if __name__ == "__main__":
    main()


