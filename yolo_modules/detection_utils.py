import torch
from yolo_modules.nms import non_max_suppression


def group_detections_by_class(boxes, scores, classes, C=20):
    """Group detections into a dict mapping class_idx -> list of (box, score)."""
    grouped = {class_idx: [] for class_idx in range(C)}
    for box, score, class_idx in zip(boxes, scores, classes):
        # ensure box is a tensor
        if not isinstance(box, torch.Tensor):
            box = torch.tensor(box)
        grouped[int(class_idx)].append((box, float(score)))
    return grouped


def classwise_nms(grouped, iou_threshold=0.5):
    final_detections = []
    for class_idx, items in grouped.items():
        if not items:
            continue

        boxes = torch.stack([item[0] for item in items])
        scores = torch.tensor([item[1] for item in items])
        keep = non_max_suppression(boxes, scores, iou_threshold=iou_threshold)

        for kept_index in keep:
            final_detections.append((boxes[kept_index], scores[kept_index].item(), class_idx))
    return final_detections


if __name__ == "__main__":
    # quick smoke
    boxes = torch.tensor([[10.,10.,20.,20.]])
    scores = torch.tensor([0.9])
    classes = torch.tensor([1])
    g = group_detections_by_class(boxes, scores, classes, C=2)
    print(g)
    print(classwise_nms(g))
