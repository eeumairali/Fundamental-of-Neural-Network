import torch
import matplotlib.pyplot as plt

def iou_calculator(boxes1, boxes2, eps=1e-6):
    """IoU for boxes in [x1, y1, x2, y2] format.
    takes x1,y1 as top left corner and x2,y2 as bottom right corner of the box
    gives how much two boxes overlap, 0 means no overlap, 1 means perfect overlap
    
    
    Intersection = max(0, x2 - x1) × max(0, y2 - y1)
    Union = Area1 + Area2 - Intersection
    IoU = Intersection / Union
    
    """
    # step 1 ensure boxes are in the right shape [N, 4] and [M, 4]
    # N columns and M rows
    # 
    if boxes1.ndim == 1: # if it's a single box, add batch dimension [2,3] will become [1, 2, 3]
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.ndim == 1:
        boxes2 = boxes2.unsqueeze(0)

    # step 2 take max and min to find the coordinates of the intersection box
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    # step 3 take width and height
    inter_w = (x2 - x1).clamp(min=0) # clamp to ensure non-negative width
    inter_h = (y2 - y1).clamp(min=0) # clamp to ensure non-negative height
    inter_area = inter_w * inter_h # area of intersection

    # step 4 calculate area of each box
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    
    # step 5 calculate union area
    union = area1[:, None] + area2[None, :] - inter_area + eps
    return inter_area / union


if __name__ == "__main__":
    ref_box = torch.tensor([100.0, 50.0, 190.0, 130.0]) # [x1, y1, x2, y2]
    boxes = torch.tensor([
        [120.0, 80.0, 200.0, 200.0], 
        [90.0, 40.0, 150.0, 150.0], 
        [50.0, 20.0, 100.0, 80.0], 
        [200.0, 150.0, 150.0, 200.0], 
    ])

    x1,y1,x2,y2 = ref_box
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label="Reference Box", color="blue")
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label=f"Box {i+1} iou {iou_calculator(ref_box, box).item():.2f}", linestyle="--")
    plt.legend()
    plt.title("Boxes Visualization")
    plt.xlim(0, 300)
    plt.ylim(0, 300)
    plt.show()