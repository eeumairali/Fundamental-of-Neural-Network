from torch.utils.data import Dataset, DataLoader
import torch
try:
    from yolo_modules.esb import encode_single_box
except ImportError:
    from esb import encode_single_box
class DummyYOLODataset(Dataset):
    def __init__(self,S,B,C, num_samples=8, image_size=448):
        self.num_samples = num_samples
        self.image_size = image_size
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(3, self.image_size, self.image_size)
        x1 = torch.randint(40, 180, (1,)).item()
        y1 = torch.randint(40, 180, (1,)).item()
        x2 = x1 + torch.randint(60, 160, (1,)).item()
        y2 = y1 + torch.randint(60, 160, (1,)).item()
        class_idx = torch.randint(0, self.C, (1,)).item()
        target = encode_single_box(self.S, self.B, self.C, torch.tensor([x1, y1, x2, y2], dtype=torch.float32), class_idx)
        return image, target


if __name__ == "__main__":
    dataset = DummyYOLODataset(num_samples=4)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for images, targets in dataloader:
        print("Batch of images shape:", images.shape) # [batch_size, 3, 448, 448]
        print("Batch of targets shape:", targets.shape) # [batch_size, 7, 7, 30]
        break