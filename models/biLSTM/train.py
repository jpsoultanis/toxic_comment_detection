import random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.sigmoid(logits).cpu().numpy() > 0.5
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    return f1


class CBFocalLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, gamma=2.0):
        super(CBFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma

        effective_num = 1.0 - torch.pow(self.beta, samples_per_cls)
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(samples_per_cls)  # Normalize

        self.class_weights = weights.to(torch.float32)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-4, 1 - 1e-4)

        # Move weights to same device as logits
        weights = self.class_weights.to(logits.device).unsqueeze(0)

        focal_weight = torch.pow(1.0 - probs * targets - (1 - probs) * (1 - targets), self.gamma)

        bce = - (targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        loss = weights * focal_weight * bce

        return loss.mean()
