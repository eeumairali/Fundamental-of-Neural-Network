import torch
import torch.nn.functional as F
from yolo_modules.cbc import corner_box_calculator
from yolo_modules.iou import iou_calculator

def yolo_v1_loss(S, B, C, LAMBDA_COORD, LAMBDA_NOOBJ, preds, targets):
    """Simplified YOLOv1 loss for predictions shaped [N, 7, 7, 30].
    takes preds and target
    returns loss for batch"""
    pred_boxes = preds[..., :10].view(-1, S, S, B, 5) # reshape to separate the two boxes and their attributes
    pred_classes = preds[..., 10:] # class probabilities for each cell ... differnt than : as ... means all dimensions except the last one

    target_boxes = targets[..., :4] # take x y w and h from target, ignore the rest for box loss calculation
    target_obj = targets[..., 4] # objectness score from target, 1 if object present, 0 if not
    target_classes = targets[..., 10:] # class probabilities from target, one-hot encoded for the true class

    obj_mask = target_obj > 0
    noobj_mask = ~obj_mask

    if obj_mask.sum() == 0:
        loss_noobj = (pred_boxes[..., 4] ** 2).mean()
        return LAMBDA_NOOBJ * loss_noobj

    pred_obj = pred_boxes[obj_mask]
    target_box = target_boxes[obj_mask]

    iou1 = iou_calculator(corner_box_calculator(pred_obj[:, 0, :4]), corner_box_calculator(target_box)).diag()
    iou2 = iou_calculator(corner_box_calculator(pred_obj[:, 1, :4]), corner_box_calculator(target_box)).diag()
    best_box_mask = iou2 > iou1

    responsible_boxes = pred_obj[torch.arange(pred_obj.size(0)), best_box_mask.long()]
    other_boxes = pred_obj[torch.arange(pred_obj.size(0)), (1 - best_box_mask.long())]
    best_iou = torch.where(best_box_mask, iou2, iou1).detach()

    coord_loss = F.mse_loss(responsible_boxes[:, :4], target_box, reduction="sum")
    obj_loss = F.mse_loss(responsible_boxes[:, 4], best_iou, reduction="sum")
    noobj_loss_pred = F.mse_loss(other_boxes[:, 4], torch.zeros_like(other_boxes[:, 4]), reduction="sum")
    noobj_loss_cells = F.mse_loss(pred_boxes[noobj_mask][..., 4], torch.zeros_like(pred_boxes[noobj_mask][..., 4]), reduction="sum")
    class_loss = F.mse_loss(pred_classes[obj_mask], target_classes[obj_mask], reduction="sum")

    total = (
        LAMBDA_COORD * coord_loss
        + obj_loss
        + LAMBDA_NOOBJ * (noobj_loss_pred + noobj_loss_cells)
        + class_loss
    )
    return total / preds.size(0) # difference that decides how backprogram will update the weights of the model, we divide by batch size to get average loss per sample

if __name__ == "__main__":

    S, B, C = 7, 2, 20
    LAMBDA_COORD = 5.0 # means 5 times important
    LAMBDA_NOOBJ = 0.5 # means 0.5 times important
    # constancts decieded by the authors to balance the different components of the loss function, they are hyperparameters that can be tuned based on the dataset and training dynamics
    print("high difference")
    preds = torch.rand(4, S, S, B * 5 + C) # batch of 4
    targets = torch.rand(4, S, S, B * 5 + C)
    loss = yolo_v1_loss(S, B, C, LAMBDA_COORD, LAMBDA_NOOBJ, preds, targets)
    print(f"for {preds}\n and {targets}\nLoss: {loss.item()}")


    # Test with close predictions
    print("\nlow difference")
    preds = torch.rand(4, S, S, B * 5 + C) * 0.1 + 0.45 # close to 0.5
    targets = torch.rand(4, S, S, B * 5 + C) * 0.1 + 0.45 # close to 0.5
    loss = yolo_v1_loss(S, B, C, LAMBDA_COORD, LAMBDA_NOOBJ, preds, targets)
    print(f"for {preds}\n and {targets}\nLoss: {loss.item()}")