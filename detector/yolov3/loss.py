"""YOLOv3 loss: objectness + no-object + box regression + classification.

Two corrections to the version written in the video:

1. It assigned into `predictions[..., 1:3]` and `target[..., 3:5]` in place.
   Writing into `target` mutates the batch tensor the dataloader handed over --
   harmless while each batch is used once, but it silently corrupts anything
   that reuses the same tensor (e.g. computing metrics on the same batch), and
   the in-place write on `predictions` is an autograd hazard the moment
   anything else reads that view. Both are computed out of place here.

2. Every term is guarded against an empty `obj` mask. An image (or, more often
   at 13x13, a whole batch at one scale) can legitimately contain no assigned
   anchor; MSE and cross-entropy over an empty tensor return NaN, and one NaN
   poisons every weight in the model on the next backward pass. This is not
   hypothetical on aerial data, where nearly all targets land on the fine scale
   and the coarse head sees empty batches routinely.
"""
import torch
import torch.nn as nn

from yolov3.utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, lambda_class=1, lambda_noobj=10, lambda_obj=1, lambda_box=10):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # How much each part of the loss is worth.
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box

    def forward(self, predictions, target, anchors):
        # target[..., 0] == -1 marks "ignore": neither an object nor a negative.
        obj = target[..., 0] == 1     # Iobj_i in the paper
        noobj = target[..., 0] == 0   # Inoobj_i
        zero = torch.zeros((), device=predictions.device, dtype=predictions.dtype)

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        no_object_loss = (
            self.bce(predictions[..., 0:1][noobj], target[..., 0:1][noobj])
            if noobj.any()
            else zero
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        # The objectness target is the IoU between the predicted box and the
        # ground truth, not a flat 1: a cell that is responsible but predicting
        # badly should not be told it is confident.
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        if obj.any():
            box_preds = torch.cat(
                [
                    self.sigmoid(predictions[..., 1:3]),
                    torch.exp(predictions[..., 3:5]) * anchors,
                ],
                dim=-1,
            )
            ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
            object_loss = self.bce(predictions[..., 0:1][obj], ious * target[..., 0:1][obj])
        else:
            object_loss = zero

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # x, y are squashed into the cell; w, h are regressed in log space
        # relative to the anchor, which is the inverse of exp(t) * anchor above.
        if obj.any():
            pred_box = torch.cat(
                [self.sigmoid(predictions[..., 1:3]), predictions[..., 3:5]], dim=-1
            )
            target_wh = torch.log(1e-16 + target[..., 3:5] / anchors)
            target_box = torch.cat([target[..., 1:3], target_wh], dim=-1)
            box_loss = self.mse(pred_box[obj], target_box[obj])
        else:
            box_loss = zero

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = (
            self.entropy(predictions[..., 5:][obj], target[..., 5][obj].long())
            if obj.any()
            else zero
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
