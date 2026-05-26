import torch.nn as nn

class ConvBlock(nn.Module):
    """backbone general class
    takes in_channels, out_channels, kernel_size, stride, and padding as arguments to create a convolutional block with Conv2d, BatchNorm2d, and LeakyReLU activation.
    return a sequential block of these layers to be used in the backbone of the network"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)
    




class YOLOv1(nn.Module):
    def __init__(self, split_size, num_boxes, num_classes):
        super().__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        self.features = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(192, 128, kernel_size=1), # 1 kernal just reduce dimentionality mainly to save computation
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)


if __name__ == "__main__":
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20)
    print(model)