"""Add independent OSNet appearance features to cached local-stage crops."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import torch

from compare_reid import load_osnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    weights = Path(__file__).resolve().parent / "weights"
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    _, encode = load_osnet(weights / "torchreid_osnet.py", weights / "osnet_x0_25_msmt17.pth",
                          ov.Core(), {"INFERENCE_NUM_THREADS": 4, "INFERENCE_PRECISION_HINT": "f32"})
    args.output.mkdir(parents=True)
    for path in sorted(args.source.glob("*/observations.jsonl")):
        target = args.output / path.parent.name
        target.mkdir()
        count = 0
        with (target / "observations.jsonl").open("w", encoding="utf-8") as out:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("embedding") and row.get("crop_ref"):
                    image = cv2.imread(str(Path(row["artifact_root"]) / row["crop_ref"]))
                    if image is None:
                        raise ValueError("Missing source crop")
                    feature = encode(image)
                    row["embedding_osnet"] = (feature / np.linalg.norm(feature)).tolist()
                    count += 1
                out.write(json.dumps(row) + "\n")
        print(path.parent.name, count, flush=True)


if __name__ == "__main__":
    main()
