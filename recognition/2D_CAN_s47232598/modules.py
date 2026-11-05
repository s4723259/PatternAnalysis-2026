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


#♥=========== Multi-scale Context Aggregation Network (CAN) ===========♥#
# TODO: Add comment here with reference, little explainer of everything

def conv_bn_relu(in_channels, out_channels, kernel_size=3, dilation=1):
    """Conv - BN - ReLU as a list of layers. 'Same' padding, so H and W are unchanged"""
    padding = dilation * (kernel_size // 2)
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]

@torch.no_grad()
def identity_init_(conv, channels, noise_ratio=0.0):
    """
    Identity initialisation of the context module, Eq. 4 (equal widths) and Eq. 5 (changing widths).

    Only the centre tap of each filter is non-zero. Input channel a feeds output channel b when they
    belong to the same group: floor(a*C / c_in) == floor(b*C / c_out), with weight C / c_out. Every
    other weight is small Gaussian noise (sigma = noise_ratio * C / c_out) which breaks ties between
    the copies of a channel. With noise_ratio = 0 and non-negative inputs the module is an exact
    pass-through at the start of training.

    :param conv: nn.Conv2d to initialise (in place)
    :param channels: C, the width of the module's input and output
    :param noise_ratio: sigma relative to the identity weight; the paper needs sigma << C / c_out
    """
    c_out, c_in, k_h, k_w = conv.weight.shape
    device = conv.weight.device

    group_in = torch.div(torch.arange(c_in, device=device) * channels, c_in, rounding_mode="floor")
    group_out = torch.div(torch.arange(c_out, device=device) * channels, c_out, rounding_mode="floor")
    same_group = group_out[:, None] == group_in[None, :]  # (c_out, c_in)

    weight = torch.randn_like(conv.weight) * (noise_ratio * channels / c_out)
    identity = torch.full((c_out, c_in), channels / c_out, dtype=weight.dtype, device=device)
    centre = weight[:, :, k_h // 2, k_w // 2]
    weight[:, :, k_h // 2, k_w // 2] = torch.where(same_group, identity, centre)

    conv.weight.copy_(weight)
    if conv.bias is not None:
        conv.bias.zero_()

class FrontEnd(nn.Module):
    """
    Dense-prediction front end (Section 4 of the paper, scaled down and trained from scratch).

    VGG-16 layout with the last THREE poolings removed (the paper removes two and works at 1/8
    resolution; thin structures such as CSF and gyral folds need the finer 1/4 resolution). Each
    removed pooling doubles the dilation of every later convolution: 1 -> 2 -> 4.

    It ends in a ReLU, so its C output maps are non-negative, which is what makes the context
    module's identity initialisation exact.

    :param in_channels: 1 for greyscale MRI slices
    :param out_channels: C, the number of feature maps passed to the context module
    :param base_channels: width of the first stage (widths: c, 2c, 4c, 8c, 8c)
    """

    def __init__(self, in_channels=1, out_channels=16, base_channels=32):
        super().__init__()
        c = base_channels
        layers = []

        # Stage 1 and 2: ordinary convolutions, each followed by a 2x2 max pool (stride 4 in total)
        layers += conv_bn_relu(in_channels, c) + conv_bn_relu(c, c)
        layers.append(nn.MaxPool2d(kernel_size=2))
        layers += conv_bn_relu(c, c * 2) + conv_bn_relu(c * 2, c * 2)
        layers.append(nn.MaxPool2d(kernel_size=2))

        # Stage 3: 3 convs, dilation 1 (stage 3's pooling removed)
        layers += conv_bn_relu(c * 2, c * 4) + conv_bn_relu(c * 4, c * 4) + conv_bn_relu(c * 4, c * 4)
        # Stage 4: 3 convs, dilation 2
        layers += conv_bn_relu(c * 4, c * 8, dilation=2)
        layers += conv_bn_relu(c * 8, c * 8, dilation=2) + conv_bn_relu(c * 8, c * 8, dilation=2)
        # Stage 5: 3 convs, dilation 4
        layers += conv_bn_relu(c * 8, c * 8, dilation=4)
        layers += conv_bn_relu(c * 8, c * 8, dilation=4) + conv_bn_relu(c * 8, c * 8, dilation=4)

        # 1x1 conv down to C feature maps
        layers += conv_bn_relu(c * 8, out_channels, kernel_size=1)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ContextModule(nn.Module):
    """
    Context module from Table 1 of the paper. Takes C feature maps and returns C feature maps of
    the same size, so it can be plugged in at any resolution.

    Layers 1-7 are 3x3 convs + ReLU with dilations 1, 1, 2, 4, 8, 16, 1. Layer 8 is a 1x1 conv with
    no ReLU. Receptive field of the 7 dilated layers: 67x67. On the front end's 64x64 maps that is
    larger than the whole map, so every output pixel can see the entire slice.

    :param channels: C
    :param variant: 'basic' (every layer C wide, ~64*C^2 parameters) or
                    'large' (widths 2C, 2C, 4C, 8C, 16C, 32C, 32C, then C)
    :param noise_ratio: tie-breaking noise for the 'large' identity init (Eq. 5). None = paper
                        setting: 0 for 'basic' (Eq. 4 has no noise), 0.01 for 'large'
    """

    DILATIONS = (1, 1, 2, 4, 8, 16, 1)
    WIDTH_MULTIPLIERS = {
        "basic": (1, 1, 1, 1, 1, 1, 1),
        "large": (2, 2, 4, 8, 16, 32, 32),
    }

    def __init__(self, channels, variant="large", noise_ratio=None):
        super().__init__()
        if variant not in self.WIDTH_MULTIPLIERS:
            raise ValueError(f"variant must be one of {list(self.WIDTH_MULTIPLIERS)}, got '{variant}'")
        if noise_ratio is None:
            noise_ratio = 0.0 if variant == "basic" else 0.01

        layers = []
        in_ch = channels
        for dilation, multiplier in zip(self.DILATIONS, self.WIDTH_MULTIPLIERS[variant]):
            out_ch = channels * multiplier
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
                       nn.ReLU(inplace=True)]
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, channels, kernel_size=1)) # Layer 8: no truncation
        self.net = nn.Sequential(*layers)

        for module in self.net.modules():
            if isinstance(module, nn.Conv2d):
                identity_init_(module, channels, noise_ratio)

    def forward(self, x):
        return self.net(x)

class CAN2D(nn.Module):
    """
    2D Context Aggregation Network: front end -> context module -> 1x1 classifier -> bilinear upsample.

    :param in_channels: 1 for greyscale MRI slices
    :param num_classes: number of output classes (4 for OASIS: background, CSF, GM, WM)
    :param base_channels: width of the first front-end stage
    :param feature_channels: C, width of the feature maps entering the context module
    :param context: 'none' (front end only, the control), 'basic' or 'large'
    :param noise_ratio: passed to ContextModule

    Two-stage training as in the paper: (1) train with context='none'; (2) build with context='large'
    (or 'basic'), load the stage-1 weights, call freeze_frontend() and train only the context module.
    Because the context module starts as an identity, stage 2 starts exactly where stage 1 finished.

    Returns raw logits of shape (B, num_classes, H, W).
    """

    def __init__(self, in_channels=1, num_classes=4, base_channels=32, feature_channels=16,
                 context="large", noise_ratio=None):
        super().__init__()
        self.frontend = FrontEnd(in_channels, feature_channels, base_channels)
        if context == "none":
            self.context = nn.Identity()
        else:
            self.context = ContextModule(feature_channels, context, noise_ratio)
        self.classifier = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        self._frontend_frozen = False

    def freeze_frontend(self):
        """Stop training the front end and classifier. BatchNorm is kept in eval mode too."""
        for module in (self.frontend, self.classifier):
            for p in module.parameters():
                p.requires_grad = False
        self._frontend_frozen = True
        self.frontend.eval()

    def train(self, mode=True):
        super().train(mode)
        if self._frontend_frozen: # Otherwise BN running statistics would still change
            self.frontend.eval()
        return self

    def forward(self, x):
        size = x.shape[-2:]
        x = self.frontend(x)
        x = self.context(x)
        x = self.classifier(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


#♥=========== Model builder ===========♥#
def build_model(arch, num_classes=4, context="large"):
    """
    :param arch: 'unet' or 'can'
    :param num_classes:
    :param context: only used by can: 'none', 'basic' or 'large'
    :return:
    """
    if arch == "unet":
        return UNet2D(in_channels=1, num_classes=num_classes, base_channels=64)
    if arch == "can":
        return CAN2D(in_channels=1, num_classes=num_classes, context=context)
    raise ValueError(f"arch must be 'unet' or 'can', got '{arch}'")