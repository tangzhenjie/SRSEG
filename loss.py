import torch

class FALoss(torch.nn.Module):
    def __init__(self):
        super(FALoss, self).__init__()
        self.FA_loss = "Module"

    def forward(self, feature1, feature2):
        return None