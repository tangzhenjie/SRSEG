import torch

class FALoss(torch.nn.Module):
    def __init__(self):
        super(FALoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.downsample = torch.nn.Upsample(scale_factor=0.125)
    def forward(self, feature1, feature2):
        feature1 = self.downsample(feature1)
        feature2 = self.downsample(feature2)
        n, c, h, w = feature1.shape
        nomal_L2 = torch.sqrt(torch.sum(torch.pow(feature1, 2)))
        feature1_A = (feature1 / nomal_L2).reshape([n, c, h * w]).permute(0, 2, 1)
        feature1_B = (feature1 / nomal_L2).reshape([n, c, h * w])
        feature1_S = torch.matmul(feature1_A, feature1_B)

        feature2_nomal_L2 = torch.sqrt(torch.sum(torch.pow(feature2, 2)))
        feature2_A = (feature2 / feature2_nomal_L2).reshape([n, c, h * w]).permute(0, 2, 1)
        feature2_B = (feature2 / feature2_nomal_L2).reshape([n, c, h * w])
        feature2_S = torch.matmul(feature2_A, feature2_B)
        return self.l1_loss(feature1_S, feature2_S)