import torch
import numpy as np

def encode_single_box(S, B, C, box_xyxy, class_idx, image_size=448):
    """Create one YOLOv1 target tensor for a single object.
    takes a box and its location with image size
    returns target for box fomrat: [objectness, x, y, w, h, class probabilities...]"""
    target = torch.zeros(S, S, B * 5 + C)

    x1, y1, x2, y2 = box_xyxy
    x1 = x1 / image_size
    y1 = y1 / image_size
    x2 = x2 / image_size
    y2 = y2 / image_size

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    cell_x = min(S - 1, int(cx * S))
    cell_y = min(S - 1, int(cy * S))

    target[cell_y, cell_x, 0:4] = torch.tensor([cx * S - cell_x, cy * S - cell_y, w, h])
    target[cell_y, cell_x, 4] = 1.0
    target[cell_y, cell_x, 10 + class_idx] = 1.0
    return target


if __name__ == "__main__":
    from cbc import corner_box_calculator
    box_xyxy = torch.tensor([120.0, 80.0, 280.0, 260.0]) # [x1, y1, x2, y2]
    S, B, C = 7, 2, 2 # Grid, Bounding Boxes per grid cell, number of classes
    target = encode_single_box(S, B, C, box_xyxy, class_idx=1) # 0 for cat, 1 for dog
    print(target.shape) # should be [7, 7, 14]
    print(f"total neurons of network: {np.prod(target.shape)}")



