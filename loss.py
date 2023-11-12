import torch
import torch.nn as nn
import torch.nn.functional as function

class CenterNetLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, lambda_size=0.1, lambda_off=1):
        super(CenterNetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_size = lambda_size
        self.lambda_off = lambda_off

    def forward(self, pred, target):
        hm_pred, wh_pred, offset_pred = pred
        hm_target, wh_target, offset_target, offset_mask = target

        loss_k = self.focal_loss(hm_pred, hm_target)  # Use focal loss
        # loss_k = self.cross_entropy(hm_pred, hm_target)  # Use cross entropy
        # loss_k = self.MSE(hm_pred, hm_target)  # Use MSE
        loss_off = self.lambda_off * self.l1_loss(offset_pred, offset_target, offset_mask)
        loss_size = self.lambda_size * self.l1_loss(wh_pred, wh_target, offset_mask)

        loss = loss_k + loss_size + loss_off
        return loss, loss_k, loss_off, loss_size

    def focal_loss(self, pred, target):
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        pos_ind = target.eq(1).float()
        neg_ind = target.lt(1).float()

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_ind
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * torch.pow(1 - target, self.beta) * neg_ind

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        num_point = pos_ind.float().sum()
        loss = - (pos_loss + neg_loss) / num_point if num_point != 0 else - neg_loss

        return loss

    def cross_entropy(self, pred, target):
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        # pred_total = pred.sum()
        # target_total = target.sum()
        # p = pred / pred_total
        # t = target / target_total
        #
        # ce_loss = t * torch.log(p) + (1 - t) * torch.log(1 - p)
        # loss = - ce_loss.sum()

        pos_ind = target.gt(0).float()
        pos_loss = pos_ind * torch.log(pred)
        pos_loss = pos_loss.sum()

        neg_ind = target.eq(0).float()
        neg_loss = neg_ind * torch.log(1 - pred)
        neg_loss = neg_loss.sum()

        point_ind = target.eq(1).float()
        num_point = pos_ind.float().sum()
        loss = - (pos_loss + neg_loss) / num_point if num_point != 0 else - neg_loss

        return loss

    def MSE(self, pred, target):
        return torch.pow(pred - target, 2).sum()

    def l1_loss(self, pred, target, mask):
        expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)
        loss = function.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

if __name__ == "__main__":
    criterion = CenterNetLoss()

    torch.random.manual_seed(100)
    sample_p = torch.rand(8, 128, 128, 2)
    sample_p = sample_p * 255

    torch.random.manual_seed(123)
    sample_t = torch.rand(8, 128, 128, 2)
    sample_t = sample_t * 255

    torch.random.manual_seed(246)
    mask = torch.rand(8, 128, 128)
    mask = mask.gt(0.8).int()

    print(criterion.l1_loss(sample_p, sample_t, mask))

