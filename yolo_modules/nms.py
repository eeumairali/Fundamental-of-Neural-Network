import torch
import matplotlib.pyplot as plt
from yolo_modules.iou import iou_calculator

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """Pure PyTorch NMS for boxes in [x1, y1, x2, y2] format.
    takes box(x1,y1,x2,y2) and scores for each box, and an IoU threshold to determine when to suppress boxes.
    returns the indices of the boxes to keep after NMS."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    keep = []
    order = scores.argsort(descending=True)
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break

        remaining = order[1:]
        ious = iou_calculator(boxes[current], boxes[remaining]).squeeze(0)
        order = remaining[ious <= iou_threshold]

    return torch.stack(keep)

if __name__ == "__main__":
    from yolo_modules.cbc import corner_box_calculator
    boxes_xywh = torch.tensor([
        [100.0, 50.0, 90.0, 80.0], 
        [101.0, 51.0, 90.0, 80.0],
        [105.0, 55.0, 90.0, 80.0],
        [120.0, 80.0, 80.0, 120.0],
        [200.0, 150.0, 50.0, 50.0],
    ]) # [x_center, y_center, width, height]
    boxes = corner_box_calculator(boxes_xywh)

    ref_box = torch.tensor([100.0, 50.0, 190.0, 130.0]) # [x1, y1, x2, y2]
    scores_model_learn = torch.tensor([0.9, 0.85, 0.8, 0.7, 0.6]) # confidence scores for each box from the model

     # [x1, y1, x2, y2]
    keep_indices = non_max_suppression(boxes, scores_model_learn)
    print("Boxes to keep after NMS:", keep_indices)


    plt.figure(figsize=(8, 5), dpi=100)
    plt.subplot(1, 2, 1)
    for box in boxes:
        x1,y1,x2,y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label=f"Box {box.tolist()}", linestyle="--")
    plt.legend()
    plt.title("Boxes Before NMS")
    plt.xlim(0, 300)
    plt.ylim(0, 300)

    # boxes after NMS
    plt.subplot(1, 2, 2)
    for idx in keep_indices:
        box = boxes[idx]
        x1,y1,x2,y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label=f"Box {box.tolist()}", linestyle="--")
    plt.legend()
    plt.title("Boxes After NMS")
    plt.xlim(0, 300)
    plt.ylim(0, 300)
    plt.show()
