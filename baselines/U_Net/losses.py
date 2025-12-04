# dice loss

def dice_loss(pred, target):
    smooth = 0.1

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    loss = ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)).mean()

    return 1 - loss

