import torch
import torch.nn as nn
import networks

class SRSEGNet(nn.Module):
    def __init__(self, num_classes, up_scale):
        super(SRSEGNet, self).__init__()

        # Encoder Network
        self.Encoder = networks.Encoder(is_restore_from_imagenet=True, resnet_weight_path="./resnetweight/")
        # Semantic Segmentation Branch
        self.SegBranch = networks.SegBranch(2048, up_scale, num_classes)

        # Super Resolution Branch
        self.SRBranch = networks.SRBranch(2048, up_scale)
        # Feature Affinity Component
        self.FABranch = nn.Sequential(
                    nn.Conv2d(2048 // up_scale, 2048 // up_scale, kernel_size=1),
                    nn.BatchNorm2d(2048 // up_scale),
                    nn.ReLU(inplace=True)
                )

    def forward(self, img_lr):
        # forward the data
        backbone =  self.Encoder(img_lr)
        # get the high-resolution image, the segmentation map and two features for FA
        feature_seg, seg_pre = self.SegBranch(backbone)
        feature_sr, img_sr = self.SRBranch(backbone)
        feature_seg_FA = self.FABranch(feature_seg)
        return img_sr, seg_pre, feature_sr, feature_seg_FA
