# Containing the source code of the components of the model. Each component must
# be implemented as a class or a function in PyTorch or TensorFlow/Keras.

# Note: Implementation must not depend on NumPy or any other library not part of 
#       PyTorch / TensorFlow unless approved!

#♥=========== Imports ===========♥#
import torch
import torch.nn as nn
import torch.nn.functional as F

#♥=========== Building blocks ===========♥#
class DoubleConv(nn.Module):
    """(Conv 3x3 - BN - ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """
    Encoder step: 2x2 max pool (halves H, W) then DoubleConv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    """
    Decoder step: transposed conv (doubles H, W halves channels), concatenate with the encoder skip
    connection, then DOuble Conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels) # in_channels // 2 (up) + in_channels//2 (skip)

    def forward(self, x, skip):
        x = self.up(x)

        # Guard against odd input sizes so that the skip connection will always line up
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


#♥=========== 2D U-Net (baseline model) ===========♥#
class UNet2D(nn.Module):
    """
    Standard 2D U-Net (Ronneberger et al., 2015) with 4 down/up-sampling stages.

    :param in_channels: 1 for greyscale MRI slices
    :param num_classes: number of output classes (4 for OASIS: background, CSF, GM, WM)
    :param base_channels: width of the first stage (64 = original U-Net, 32 = lighter)

    Returns raw logits of shape (B, num_classes, H, W); softmax is applied inside the loss
    / at prediction time.
    """

    def __init__(self, in_channels=1, num_classes=4, base_channels=64):
        super().__init__()
        c = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16) # Bottleneck

        # Decoder
        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)

        # 1x1 conv to class logits
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)
