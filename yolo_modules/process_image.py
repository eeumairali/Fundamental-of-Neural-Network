import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def normalize_resize_torch_conversion(np_img, target_size=(448, 448), show_info=False, plot_img=False):
    """function will permute (H, W, C) to (1, C, H, W)
    then apply bilinear interpolation to resize to (448, 448)
    align_corners false for pixel center based scalling no distortion"""

    normalized_img = torch.tensor(np_img, dtype=torch.float32) / 255.0 # normalize to [0, 1]
    permuted = normalized_img.permute(2, 0, 1).unsqueeze(0) # reorder [H, W, C] to [None, C, H, W]
    interpolated = F.interpolate(permuted, size=target_size, mode="bilinear", align_corners=False) # 
    if show_info:
        print(f"preprocessed image shape: {interpolated.shape}")
    if plot_img:
        plt.imshow(interpolated.squeeze().permute(1, 2, 0).numpy())
        plt.title("Preprocessed Image")
        plt.show()
    return interpolated