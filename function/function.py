import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm

def seed_func():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, output, target):
        upper = torch.exp(output[:, target])
        lower = torch.exp(output).sum(1)
        loss = -torch.log(upper / lower)
        return loss

def cal_accuracy_fewshot_1shot(loader, net, device):
    true_label = 0
    num_batches = 0

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = support_images.permute(1, 0, 2, 3, 4).to(device)
        targets = query_targets.to(device)
        targets = targets.permute(1, 0)

        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            true_label += 1 if torch.argmax(scores) == target else 0
            num_batches += 1

    return true_label / num_batches

def convert_for_5shots(support_images, support_targets, device):
    support_targets = support_targets.cpu()
    labels = torch.unique(support_targets)
    new_support_images = []

    for label in labels:
        label_images = support_images[:, support_targets[0] == label]
        padded_label_images = torch.zeros((5, 3, 224, 224), dtype=label_images.dtype)
        padded_label_images[:label_images.shape[1]] = label_images.squeeze(0)
        new_support_images.append(padded_label_images.to(device))

    return new_support_images

def cal_accuracy_fewshot_5shot(loader, net, device):
    true_label = 0
    num_batches = 0

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = convert_for_5shots(support_images, support_targets, device)
        targets = query_targets.to(device)
        targets = targets.permute(1, 0)

        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            true_label += 1 if torch.argmax(scores) == target else 0
            num_batches += 1

    return true_label / num_batches

def predicted_fewshot_1shot(loader, net, device):
    predicted = []
    true_labels = []

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = support_images.permute(1, 0, 2, 3, 4).to(device)
        targets = query_targets.to(device)
        targets = targets.permute(1, 0)

        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            predicted.append(torch.argmax(scores).cpu().detach().numpy())
            true_labels.append(target.cpu().detach().numpy())

    return np.array(true_labels), np.array(predicted)

def predicted_fewshot_5shot(loader, net, device):
    predicted = []
    true_labels = []

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = convert_for_5shots(support_images, support_targets, device)
        targets = query_targets.to(device)
        targets = targets.permute(1, 0)

        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            predicted.append(torch.argmax(scores).cpu().detach().numpy())
            true_labels.append(target.cpu().detach().numpy())

    return np.array(true_labels), np.array(predicted)
