import torch


@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, masks: torch.Tensor) -> float:
    """
    Computes pixel accuracy.

    Args:
        logits: (B, C, H, W)
        masks : (B, H, W)

    Returns:
        float between 0 and 1
    """
    preds = torch.argmax(logits, dim=1)

    correct = (preds == masks).sum().item()
    total = masks.numel()

    return correct / total


@torch.no_grad()
def dice_score(logits: torch.Tensor,
               masks: torch.Tensor,
               num_classes: int,
               eps: float = 1e-6) -> float:
    """
    Computes mean Dice score over all classes.
    """
    preds = torch.argmax(logits, dim=1)

    dice = []

    for cls in range(num_classes):

        pred_cls = preds == cls
        mask_cls = masks == cls

        intersection = (pred_cls & mask_cls).sum().float()
        union = pred_cls.sum().float() + mask_cls.sum().float()

        score = (2 * intersection + eps) / (union + eps)
        dice.append(score)

    return torch.mean(torch.stack(dice)).item()


@torch.no_grad()
def mean_iou(logits: torch.Tensor,
             masks: torch.Tensor,
             num_classes: int,
             eps: float = 1e-6) -> float:
    """
    Computes mean Intersection over Union (mIoU).
    """
    preds = torch.argmax(logits, dim=1)

    ious = []

    for cls in range(num_classes):

        pred_cls = preds == cls
        mask_cls = masks == cls

        intersection = (pred_cls & mask_cls).sum().float()
        union = (pred_cls | mask_cls).sum().float()

        iou = (intersection + eps) / (union + eps)
        ious.append(iou)

    return torch.mean(torch.stack(ious)).item()