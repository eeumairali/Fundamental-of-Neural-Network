import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch


def draw_detections(image_tensor, detections, class_names=None):
    """Draw detections on a single image tensor in CHW format (batched: 1,C,H,W accepted).

    detections: list of (box, score, class_idx) with box=[x1,y1,x2,y2]
    """
    image = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image = image.clip(0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)

    for box, score, class_idx in detections:
        if isinstance(box, torch.Tensor):
            x1, y1, x2, y2 = box.tolist()
        else:
            x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        label = f"{class_idx} {score:.2f}" if class_names is None else f"{class_names[class_idx]} {score:.2f}"
        ax.text(x1, y1, label, color="white", bbox=dict(facecolor="black", alpha=0.5))

    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    import torch
    img = torch.rand(1,3,448,448)
    draw_detections(img, [(torch.tensor([10.,10.,20.,20.]), 0.9, 1)], class_names=["cat","dog"])    
