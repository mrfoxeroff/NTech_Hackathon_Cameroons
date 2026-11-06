import numpy as np
import pytest

from local_tracking.stabilization import GeometryStabilizer


def obs(box, local_id=1):
    x1, y1, x2, y2 = box
    return dict(
        local_track_id=local_id,
        bbox_xyxy=list(box),
        footpoint_px=[(x1 + x2) / 2, y2],
        touches_frame_bottom=False,
        footpoint_method="bbox_bottom_center",
    )


def stabilizer():
    return GeometryStabilizer(400, 400, [[[0, 0.3], [1, 0.3], [1, 0.7], [0, 0.7]]])


def test_occluded_bottom_does_not_jump_and_raw_geometry_remains():
    st = stabilizer()
    st.update([obs([100, 20, 180, 300])], 0.0)
    o = obs([102, 20, 182, 190])
    st.update([o], 0.04)
    assert o["bbox_xyxy"] == [102, 20, 182, 190]
    assert o["raw_footpoint_px"] == [142, 190]
    assert o["footpoint_status"] == "estimated"
    assert o["footpoint_px"][1] == pytest.approx(300)
    assert o["bbox_stabilized_xyxy"][3] == pytest.approx(300)


def test_prediction_is_bounded_and_recovers_when_legs_visible():
    st = stabilizer()
    st.update([obs([100, 20, 180, 300])], 0.0)
    for t in np.arange(0.04, 1.21, 0.04):
        o = obs([100, 20, 180, 180])
        st.update([o], float(t))
    assert o["footpoint_status"] == "unknown" and o["footpoint_px"] is None
    recovered = obs([100, 20, 180, 300])
    st.update([recovered], 1.24)
    assert recovered["footpoint_status"] == "observed"
    assert recovered["footpoint_px"][1] == pytest.approx(300)


def test_histories_are_independent_and_gap_resets_geometry():
    st = stabilizer()
    st.update([obs([100, 20, 180, 300]), obs([200, 30, 240, 130], 2)], 0.0)
    a = obs([100, 20, 180, 180])
    b = obs([200, 30, 240, 130], 2)
    st.update([a, b], 0.04)
    assert a["footpoint_status"] == "estimated"
    assert b["footpoint_status"] == "observed"
    reset = obs([100, 20, 180, 180])
    st.update([reset], 2.0)
    assert reset["footpoint_status"] == "observed"
    assert reset["footpoint_px"][1] == 180


def test_bottom_clipped_at_first_observation_has_unknown_feet():
    o = obs([100, 20, 180, 400])
    o["touches_frame_bottom"] = True
    stabilizer().update([o], 0.0)
    assert o["footpoint_px"] is None and o["footpoint_status"] == "unknown"


def test_no_occluder_does_not_turn_every_height_change_into_prediction():
    st = GeometryStabilizer(400, 400)
    st.update([obs([100, 20, 180, 300])], 0.0)
    o = obs([100, 20, 180, 150])
    st.update([o], 0.04)
    assert o["footpoint_status"] == "observed"


def test_real_table_occlusion_example_does_not_jump_by_352_pixels():
    st = GeometryStabilizer(
        2592,
        1944,
        [
            [
                [345 / 1296, 306 / 972],
                [800 / 1296, 247 / 972],
                [944 / 1296, 416 / 972],
                [378 / 1296, 538 / 972],
            ]
        ],
    )
    first = obs([911.25, 603.45, 1358.775, 1684.8], 3)
    second = obs([907.2, 595.35, 1358.775, 1332.45], 3)
    st.update([first], 0.0)
    st.update([second], 0.04)
    assert abs(second["raw_footpoint_px"][1] - first["raw_footpoint_px"][1]) > 350
    assert abs(second["footpoint_px"][1] - first["footpoint_px"][1]) < 30
    assert second["footpoint_status"] == "estimated"


def test_hanging_lamp_upper_body_occlusion_preserves_visible_feet():
    st = GeometryStabilizer(400, 400, [[[0, 0.32], [1, 0.32], [1, 0.4], [0, 0.4]]])
    st.update([obs([100, 20, 180, 300])], 0.0)
    lower_body = obs([100, 160, 180, 300])
    st.update([lower_body], 0.04)
    assert lower_body["bbox_stabilization_status"] == "estimated"
    assert lower_body["bbox_stabilized_xyxy"][1] == pytest.approx(20)
    assert lower_body["footpoint_px"][1] == pytest.approx(300)
    assert lower_body["footpoint_status"] == "observed"
    assert "upper_body_occluded" in lower_body["geometry_flags"]
