from tensorflow.keras import backend as k


# <-- Dice Loss -->
def dice_coef(y_true, y_pred):
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    intersection = k.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0001) / (k.sum(y_true_f) + k.sum(y_pred_f) + 0.0001)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# <-- Tversky Loss-->
def tversky(y_true, y_pred):
    smooth = 1
    y_true_pos = k.flatten(y_true)
    y_pred_pos = k.flatten(y_pred)
    true_pos = k.sum(y_true_pos * y_pred_pos)
    false_neg = k.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = k.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)
