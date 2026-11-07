"""Compare Intel 0288 and Torchreid OSNet on the same visually labeled pilot crops."""

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import av
import cv2
import numpy as np
import openvino as ov
import torch
from torchvision import transforms

from associate import DEFAULTS, Observation, associate, split_tracklets, synchronized_pairs


# The two trajectories were visually checked in the pilot. These labels are
# used only for evaluation and are never passed to the association algorithm.
PILOT_LABELS = {("cam_1", 2): 1, ("cam_1", 5): 2,
                ("cam_2", 1): 1, ("cam_2", 3): 2, ("cam_2", 20): 1}


def load_osnet(source, checkpoint, core, cpu_options):
    spec = importlib.util.spec_from_file_location("torchreid_osnet", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = module.osnet_x0_25(num_classes=state["classifier.weight"].shape[0], pretrained=False)
    model.load_state_dict(state, strict=True)
    preprocess = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((256, 128)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    model.eval()
    ir = ov.convert_model(model, example_input=torch.zeros((1, 3, 256, 128)))
    compiled = core.compile_model(ir, "CPU", cpu_options)

    def tensor(crop):
        return preprocess(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))[None]

    @torch.inference_mode()
    def encode_torch(crop):
        return model(tensor(crop)).numpy().ravel()

    def encode_openvino(crop):
        return compiled([tensor(crop).numpy()])[compiled.output(0)].reshape(-1)

    return encode_torch, encode_openvino


def collect_crops(rows, cameras):
    requested = defaultdict(lambda: defaultdict(list))
    seek_by_pts = "source_pts" in rows[0]
    for index, row in enumerate(rows):
        if "embedding" in row:
            requested[row["camera_id"]][row["source_pts" if seek_by_pts else "frame_index"]].append(index)
    crops = {}
    for camera, frames in requested.items():
        with av.open(str(cameras[camera])) as container:
            if seek_by_pts:
                container.seek(min(frames), stream=container.streams.video[0], backward=True)
            for frame_index, frame in enumerate(container.decode(video=0)):
                frame_key = frame.pts if seek_by_pts else frame_index
                if frame_key > max(frames):
                    break
                if frame_key not in frames:
                    continue
                image = frame.to_ndarray(format="bgr24")
                for index in frames[frame_key]:
                    x1, y1, x2, y2 = rows[index]["bbox_xyxy"]
                    crops[index] = image[y1:y2, x1:x2].copy()
    if crops.keys() != {i for i, row in enumerate(rows) if "embedding" in row}:
        raise ValueError("Could not recover all crops from the videos")
    return crops


def evaluate(rows, embeddings, settings, reference):
    samples = list(embeddings)
    matrix = np.stack([embeddings[index] for index in samples])
    similarities = matrix @ matrix.T
    cameras = np.array([rows[index]["camera_id"] for index in samples])
    labels = np.array([reference[(rows[index]["camera_id"], rows[index]["local_id"])] for index in samples])
    cross_camera = cameras[:, None] != cameras[None, :]
    same = labels[:, None] == labels[None, :]
    ranked = np.where(cross_camera, similarities, -np.inf).argmax(axis=1)
    pairs = np.triu(cross_camera, k=1)
    positive, negative = similarities[pairs & same], similarities[pairs & ~same]
    sorted_negative = np.sort(negative)
    if len(positive) and len(negative):
        lower = np.searchsorted(sorted_negative, positive, side="left")
        upper = np.searchsorted(sorted_negative, positive, side="right")
        auc = float((lower + upper).mean() / (2 * len(negative)))
    else:
        auc = None
    queries = (cross_camera & same).any(axis=1)
    results = {"crops": len(samples), "rank1_queries": int(queries.sum()),
               "rank1_correct": int((queries & (labels == labels[ranked])).sum()),
               "auc": auc,
               "positive_pairs": len(positive), "negative_pairs": len(negative),
               "positive_median": float(np.median(positive)) if len(positive) else None,
               "negative_median": float(np.median(negative)) if len(negative) else None}

    observations = [Observation.parse(row | {"embedding": embeddings[index].tolist()}
                                     if index in embeddings else row) for index, row in enumerate(rows)]
    cfg = DEFAULTS | settings
    tracklets = split_tracklets(observations, cfg["max_local_gap_ms"])
    results["eligible_tracklet_pairs"] = []
    for a in tracklets:
        for b in tracklets:
            if a.first.camera != "cam_1" or b.first.camera != "cam_2":
                continue
            if max(a.first.time, b.first.time) > min(a.last.time, b.last.time):
                continue
            pairs_count = len(list(synchronized_pairs(a.observations, b.observations, cfg["sync_tolerance_ms"])))
            if pairs_count < cfg["min_overlap_pairs"] or a.reid is None or b.reid is None:
                continue
            results["eligible_tracklet_pairs"].append({
                "cam_1_id": a.first.local_id, "cam_2_id": b.first.local_id, "sync_pairs": pairs_count,
                "same_person": reference[(a.first.camera, a.first.local_id)] ==
                               reference[(b.first.camera, b.first.local_id)],
                "cosine": float(np.dot(a.reid, b.reid)),
            })

    results["sweep"] = {}
    for name, brightness_limit in [("appearance_only", None), ("pilot_brightness_gate", cfg["max_brightness_difference"])]:
        trials = []
        for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            groups = associate(observations, cfg | {"reid_min_similarity": threshold,
                                                    "max_brightness_difference": brightness_limit})
            group_labels = [{reference[(obs.camera, obs.local_id)] for obs in group} for group in groups]
            mixed = sum(len(group) > 1 for group in group_labels)
            label_frequency = Counter(label for group in group_labels for label in group)
            trials.append({"threshold": threshold, "total_tracks": len(groups), "mixed_tracks": mixed,
                           "split_reference_tracks": sum(count > 1 for count in label_frequency.values()),
                           "correct": len(groups) == len(set(reference.values())) and mixed == 0})
        results["sweep"][name] = trials
    return results


def benchmark(rows, crops, settings, reference, models):
    results = {"observations": len(rows), "reference_tracks": len(set(reference.values())), "models": {}}
    reference_osnet = None
    for name, encode in models.items():
        for crop in list(crops.values())[:6]:
            encode(crop)
        embeddings, times = {}, []
        for index, crop in crops.items():
            started = perf_counter()
            vector = encode(crop)
            times.append((perf_counter() - started) * 1000)
            embeddings[index] = vector / np.linalg.norm(vector)
        metrics = evaluate(rows, embeddings, settings, reference)
        metrics["cpu_ms_median_per_crop"] = float(np.median(times))
        metrics["cpu_ms_p90_per_crop"] = float(np.quantile(times, 0.90))
        if name.startswith("intel"):
            metrics["lowest_cosine_to_cached_embedding"] = min(
                float(np.dot(vector, rows[index]["embedding"])) for index, vector in embeddings.items())
        elif reference_osnet is None:
            reference_osnet = embeddings
        else:
            metrics["lowest_cosine_to_pytorch_embedding"] = min(
                float(np.dot(vector, reference_osnet[index])) for index, vector in embeddings.items())
        results["models"][name] = metrics
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Workspace containing the videos")
    parser.add_argument("--cases", type=Path, help="Optional benchmark manifest with paths and labels")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mtmc, weights = args.root / "mtmc", args.root / "mtmc/weights"
    settings = json.loads((mtmc / "camera_graph.pilot.json").read_text())
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    core = ov.Core()
    cpu_options = {"INFERENCE_NUM_THREADS": 4, "INFERENCE_PRECISION_HINT": "f32"}
    intel = core.compile_model(str(weights / "person-reidentification-retail-0288.xml"), "CPU", cpu_options)

    def encode_intel(crop):
        tensor = cv2.resize(crop, (128, 256)).transpose(2, 0, 1)[None].astype(np.float32)
        return intel([tensor])[intel.output(0)].reshape(-1)

    encode_osnet, encode_osnet_ov = load_osnet(weights / "torchreid_osnet.py",
                                              weights / "osnet_x0_25_msmt17.pth", core, cpu_options)
    models = {"intel_0288_openvino": encode_intel, "osnet_x0_25_msmt17_pytorch": encode_osnet,
              "osnet_x0_25_msmt17_openvino": encode_osnet_ov}
    if args.cases:
        manifest = json.loads(args.cases.read_text(encoding="utf-8"))
        cameras = {name: args.root / path for name, path in manifest["cameras"].items()}
        results = {"cameras": manifest["cameras"], "cases": {}}
        for case in manifest["cases"]:
            reference = {}
            ranges = {}
            for camera, local_id, label, *interval in case["labels"]:
                key = (camera, local_id)
                if key in reference:
                    raise ValueError(f"Duplicate local ID in {case['name']}: {key}")
                reference[key] = label
                ranges[key] = interval or case["window_ms"]
            raw = [json.loads(line) for line in (args.cases.parent / case["observations"]).read_text().splitlines()]
            rows = [r for r in raw if (key := (r["camera_id"], r["local_id"])) in ranges
                    and ranges[key][0] <= r["timestamp_ms"] < ranges[key][1]]
            if {(r["camera_id"], r["local_id"]) for r in rows} != reference.keys():
                raise ValueError(f"Missing labeled local IDs in {case['name']}")
            crops = collect_crops(rows, cameras)
            result = benchmark(rows, crops, settings, reference, models)
            results["cases"][case["name"]] = result | {"window_ms": case["window_ms"],
                                                       "unlabeled_observations": len(raw) - len(rows)}
            print(f"{case['name']}: {len(crops)} crops, {len(rows)} labeled observations", flush=True)
    else:
        rows = [json.loads(line) for line in (mtmc / "pilot-aligned/observations_clean.jsonl").read_text().splitlines()]
        crops = collect_crops(rows, {f"cam_{i}": args.root / f"camera_new_{i} - Trim.mp4" for i in (1, 2)})
        results = benchmark(rows, crops, settings, PILOT_LABELS, models)
        results["reference_people"] = results["reference_tracks"]
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
        file.write("\n")


if __name__ == "__main__":
    main()
