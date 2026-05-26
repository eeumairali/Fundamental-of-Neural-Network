import torch
from yolo_modules.decode_predictions import decode_predictions
from yolo_modules.detection_utils import group_detections_by_class, classwise_nms


def infer_and_postprocess(model, image_tensor, conf_threshold=0.25, iou_threshold=0.5, image_size=448, S=7, B=2, C=20):
    model.eval()
    with torch.no_grad():
        preds = model(image_tensor)
        boxes, scores, classes = decode_predictions(preds, image_size=image_size, conf_threshold=conf_threshold, S=S, B=B, C=C)
        if boxes.numel() == 0:
            return []
        grouped = group_detections_by_class(boxes, scores, classes, C=C)
        final_detections = classwise_nms(grouped, iou_threshold=iou_threshold)
    return final_detections


if __name__ == "__main__":
    import torch
    from yolo_modules.yolo_v1_architecture import YOLOv1
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20)
    img = torch.rand(1,3,448,448)
    print(infer_and_postprocess(model, img))
