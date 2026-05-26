
import torch
import matplotlib.pyplot as plt

def corner_box_calculator(boxes):
    """Convert [..., x, y, w, h] boxes to [..., x1, y1, x2, y2].
    takes x,y as center of the box and w,h as width and height of the box
    returns x1,y1 as top left corner and x2,y2 as bottom right corner of the box"""
    x, y, w, h = boxes.unbind(-1) # [1,2,3,4] will become x=1, y=2, w=3, h=4
    half_w = w / 2
    half_h = h / 2
    return torch.stack([x - half_w, y - half_h, x + half_w, y + half_h], dim=-1)



if __name__ == "__main__":
    boxes_xywh = torch.tensor(
        [100.0, 50.0, 90.0, 80.0], 
    ) # [x_center, y_center, width, height]
    
    x1, y1, x2, y2 = corner_box_calculator(boxes_xywh)
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label=f"Box {boxes_xywh.tolist()}", linestyle="--")
    plt.legend()
    plt.title(f"Box in xywh format: {boxes_xywh.tolist()} \nconverted to xyxy format: {[x1.item(), y1.item(), x2.item(), y2.item()]}")
    plt.xlim(0, 300)
    plt.ylim(0, 300)
    plt.show()