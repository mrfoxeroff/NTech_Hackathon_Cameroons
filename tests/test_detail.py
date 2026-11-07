import numpy as np
import torch
from ultralytics.engine.results import Results

from local_tracking.detail import refine_distant, same_detection


def test_distant_crop_masks_return_to_source_coordinates():
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    whole_mask = torch.zeros((1, 120, 180), dtype=torch.uint8)
    whole_mask[0, 20:52, 64:83] = 1
    full = Results(image, "", {0: "person"}, boxes=torch.tensor([[64, 20, 83, 52, 0.7, 0]]), masks=whole_mask)

    def predict(crop, _):
        mask = torch.zeros((1, *crop.shape[:2]), dtype=torch.uint8)
        mask[0, 20:53, 20:39] = 1
        return Results(crop, "", {0: "person"}, boxes=torch.tensor([[20, 20, 39, 53, 0.8, 0]]), masks=mask)

    result = refine_distant(full, image, [0.25, 0, 0.75, 0.65], predict)
    assert result.boxes.xyxy.tolist() == [[65, 20, 84, 53]]
    assert result.masks.data.shape == (1, 120, 180)
    assert int(result.masks.data[0, 20:53, 65:84].sum()) == 33 * 19


def test_adjacent_people_are_not_treated_as_one_distant_detection():
    assert same_detection([20, 30, 50, 105], [23, 31, 47, 74])
    assert not same_detection([20, 30, 50, 105], [35, 31, 65, 104])
