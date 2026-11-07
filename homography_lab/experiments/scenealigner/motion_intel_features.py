"""Optional second appearance descriptor using the existing local OpenVINO environment."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=WORKSPACE / "mtmc/weights")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cv2.setNumThreads(2)
    import openvino as ov

    weights = args.weights_root / "person-reidentification-retail-0288.xml"
    for path in [weights, weights.with_suffix(".bin")]:
        if not path.is_file():
            raise FileNotFoundError(path)
    compiled = ov.Core().compile_model(str(weights), "CPU", {"INFERENCE_NUM_THREADS": 4})

    def encode(crop):
        tensor = cv2.resize(crop, (128, 256)).transpose(2, 0, 1)[None].astype(np.float32)
        return compiled([tensor])[compiled.output(0)].reshape(-1)

    previous = None
    image = None
    with args.output.open("x", encoding="utf-8") as out:
        for i, line in enumerate(args.observations.read_text().splitlines()):
            row = json.loads(line)
            if row["image"] != previous:
                image = cv2.imread(row["image"])
                previous = row["image"]
            x1, y1, x2, y2 = np.asarray(row["box"]).astype(int)
            feature = np.asarray(encode(image[max(0, y1) : y2, max(0, x1) : x2]), float)
            row["embedding_intel"] = (feature / np.linalg.norm(feature)).tolist()
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
            if i % 500 == 0:
                print("Second appearance descriptor", i, flush=True)


if __name__ == "__main__":
    main()
