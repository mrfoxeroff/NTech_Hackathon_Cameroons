"""Combine a native-resolution distant crop with the full-frame detector."""

import torch
import torch.nn.functional as F


def same_detection(a, b):
    ax, ay, ar, ab = a[:4]
    bx, by, br, bb = b[:4]
    aw, ah, bw, bh = ar - ax, ab - ay, br - bx, bb - by
    overlap = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
    return (
        overlap >= 0.75 * min(aw * ah, bw * bh)
        and abs(ax + ar - bx - br) <= 0.5 * min(aw, bw)
        and abs(ay - by) <= 0.2 * min(ah, bh)
    )


def refine_distant(result, image, roi, predict):
    height, width = image.shape[:2]
    left, top, right, bottom = [round(v * scale) for v, scale in zip(roi, (width, height) * 2, strict=True)]
    if right - left < 40 or bottom - top < 40:
        return result
    zoom = predict(image[top:bottom, left:right], 0.50)
    if not len(zoom.boxes):
        return result
    source_boxes = zoom.boxes.data.clone()
    source_boxes[:, :4] += source_boxes.new_tensor([left, top, left, top])
    boxes = source_boxes.cpu().tolist()
    selected = []
    # Prefer the complete mask over a second torso box for the same subject.
    for i in sorted(range(len(boxes)), key=lambda i: boxes[i][4] * (boxes[i][3] - boxes[i][1]), reverse=True):
        x1, y1, x2, y2, *_ = boxes[i]
        if (
            x1 < left + 12
            or x2 > right - 12
            or y1 < top + 6
            or y2 > bottom - 12
            or y2 - y1 > 0.66 * (bottom - top)
            or any(same_detection(boxes[i], boxes[j]) for j in selected)
        ):
            continue
        selected.append(i)
    if not selected:
        return result
    full = result.boxes.data.cpu().tolist()
    keep = [i for i, box in enumerate(full) if not any(same_detection(box, boxes[j]) for j in selected)]
    crop_masks = F.pad(zoom.masks.data[selected], (left, width - right, top, height - bottom))
    kept_boxes = result.boxes.data[keep]
    if keep:
        crop_masks = torch.cat((result.masks.data[keep], crop_masks))
    combined = result.new()
    combined.update(boxes=torch.cat((kept_boxes, source_boxes[selected])), masks=crop_masks)
    return combined
