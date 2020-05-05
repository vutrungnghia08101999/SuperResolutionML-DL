import torch
from torch import nn

def MSELoss(preds: torch.tensor, labels: torch.tensor) -> torch.float:
    """
    preds, label - batch_size x 1 x 51 x 51 for x3 scale 
    """
    criterion = nn.MSELoss()
    return criterion(preds, labels)

def get_features(vgg, imgs: torch.tensor, vgg_depth=8) -> torch.tensor:
    """
    imgs - batch_size x 1 x 51 x 51 for x3 scale
    """
    assert imgs.shape[1] == 1
    s = imgs.repeat(1, 3, 1, 1)
    for i in range(vgg_depth):
        s = vgg.features[i](s)
    return s

def perceptual_loss(vgg, preds: torch.tensor, labels: torch.tensor, depth=8) -> torch.float:
    """
    preds, label - batch_size x 1 x 51 x 51 for x3 scale 
    """
    preds_features = get_features(vgg=vgg, imgs=preds)
    labels_features = get_features(vgg=vgg, imgs=labels)
    return MSELoss(preds=preds_features, labels=labels_features)
