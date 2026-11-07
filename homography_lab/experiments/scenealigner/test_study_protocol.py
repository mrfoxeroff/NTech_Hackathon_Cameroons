import json
import sys

import lens_structure_experiment as study
import pytest


def test_reference_labels_are_never_sent_to_inference_and_evaluation_is_last(tmp_path, monkeypatch):
    inputs, groups, reference, output = [
        tmp_path / name for name in ["inputs", "groups", "reference", "output"]
    ]
    calls = []
    monkeypatch.setattr(study, "validate_original_groups", lambda *args: None)
    monkeypatch.setattr(study.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "study",
            "--inputs",
            str(inputs),
            "--groups",
            str(groups),
            "--reference",
            str(reference),
            "--output",
            str(output),
        ],
    )
    study.main()
    inference_names = [
        "rectify_inputs.py",
        "semantic_inputs.py",
        "check_lens.py",
        "grouped_experiment.py",
        "fragment_search.py",
    ]
    inference = [i for i, call in enumerate(calls) if any(call[1].endswith(name) for name in inference_names)]
    evaluation = [i for i, call in enumerate(calls) if call[1].endswith("evaluate.py")]
    assert max(inference) < min(evaluation)
    assert all(str(reference) not in calls[i] for i in inference)
    searches = [call for call in calls if call[1].endswith("fragment_search.py")]
    assert len(searches) == 4
    assert sum("--lens-json" in call for call in searches) == 2
    assert sum("--structural-inputs" in call for call in searches) == 2


def test_study_rejects_reconstructions_from_other_camera_images(tmp_path):
    inputs, groups = tmp_path / "inputs", tmp_path / "groups"
    (inputs / "views").mkdir(parents=True)
    (inputs / "views/cam_1.jpg").write_bytes(b"new camera image")
    folder = groups / "group_01/reconstruction"
    folder.mkdir(parents=True)
    old = groups / "old.jpg"
    old.write_bytes(b"old camera image")
    old = old.rename(groups / "cam_1.jpg")
    (groups / "groups.json").write_text(json.dumps({"groups": [["cam_1"]]}))
    (folder / "manifest.json").write_text(json.dumps({"status": "complete", "images": [str(old)]}))
    with pytest.raises(ValueError, match="do not belong"):
        study.validate_original_groups(groups, inputs)
