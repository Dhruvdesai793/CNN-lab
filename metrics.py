import torch


@torch.no_grad()
def pixel_accuracy(logits, masks):
    preds = logits.argmax(dim=1)
    return (preds == masks).float().mean().item()


@torch.no_grad()
def dice_score(logits, masks, num_classes, eps=1e-6):
    preds = logits.argmax(dim=1)

    scores = []

    for cls in range(num_classes):

        pred = preds == cls
        target = masks == cls

        if not pred.any() and not target.any():
            continue

        intersection = (pred & target).sum().float()
        denominator = pred.sum().float() + target.sum().float()

        scores.append((2 * intersection + eps) / (denominator + eps))

    return torch.stack(scores).mean().item()


@torch.no_grad()
def mean_iou(logits, masks, num_classes, eps=1e-6):
    preds = logits.argmax(dim=1)

    scores = []

    for cls in range(num_classes):

        pred = preds == cls
        target = masks == cls

        if not pred.any() and not target.any():
            continue

        intersection = (pred & target).sum().float()
        union = (pred | target).sum().float()

        scores.append((intersection + eps) / (union + eps))

    return torch.stack(scores).mean().item()