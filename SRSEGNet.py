import torch
import torch.nn as nn
import networks

class SRSEGNet(nn.Module):
    def __init__(self, num_classes):
        super(SRSEGNet, self).__init__()

        # Encoder Network
        self.Encoder = networks.Encoder(is_restore_from_imagenet=True, resnet_weight_path="./resnetweight/")
        # Semantic Segmentation Branch
        self.SegBranch = networks.Classifier_Module_Mul(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.SegUpBranch =
        # Super Resolution Branch

        # Feature Affinity Component
        pass

    def forward(self, img_lr):

        # forward the data
        self.feature =  self.Encoder(img_lr)
        # get the high-resolution image, the segmentation map and two features for FA
        img_sr = None
        seg_pre = None
        feature_seg = None
        feature_sr = None
        return img_sr, seg_pre, feature_sr, feature_seg
