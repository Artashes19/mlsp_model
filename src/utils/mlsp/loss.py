import piq
import torch
import torch.nn as nn


def se(preds, targets, masks, weights=None):
    sq_error = (preds - targets) ** 2 * masks
    if weights is not None:
        sq_error *= weights
    sq_error_sum = sq_error.sum()
    return sq_error_sum


def mse(preds, targets, masks, weights):
    mask_sum = masks.sum()
    sq_error_sum = se(preds, targets, masks)
    return sq_error_sum / (mask_sum + 1e-8)


def rmse(preds, targets, masks):
    mse_loss = mse(preds, targets, masks)
    return torch.sqrt(mse_loss)


def anchored_mse(preds, targets, masks, anchor, alpha=0.1):
    main_loss = mse(preds, targets, masks)
    anchor_loss = mse(preds, anchor, masks)
    
    weighted_loss = main_loss + alpha * anchor_loss
    return weighted_loss


class CustomGradientDifferenceLoss(nn.Module):
    """
    Gradient Difference Loss implementation
    (since it's not commonly available in libraries)
    """
    
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target, mask=None):
        # Calculate gradients
        pred_dx = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_dy = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        
        target_dx = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_dy = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        # Calculate gradient differences
        grad_diff_x = torch.abs(pred_dx - target_dx) ** self.alpha
        grad_diff_y = torch.abs(pred_dy - target_dy) ** self.alpha
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask_x = mask[:, :, 1:, :]
            mask_y = mask[:, :, :, 1:]
            
            grad_diff_x = grad_diff_x * mask_x
            grad_diff_y = grad_diff_y * mask_y
            
            # Sum and normalize by mask
            loss = (grad_diff_x.sum() / (mask_x.sum() + 1e-8)) + (grad_diff_y.sum() / (mask_y.sum() + 1e-8))
        else:
            loss = torch.mean(grad_diff_x) + torch.mean(grad_diff_y)
        
        return loss


class CustomF1Loss(nn.Module):
    """
    Custom F1 Loss that works correctly with regression problems
    """
    
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target, mask=None):
        # For regression problems, we use a different approach than the standard Dice/F1
        # Calculate the "similarity" between predictions and targets
        
        # Calculate absolute difference
        diff = torch.abs(pred - target)
        
        # Scale by the max value for normalization
        max_val = torch.max(torch.max(pred), torch.max(target)) + self.epsilon
        similarity = 1.0 - (diff / max_val)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            similarity = similarity * mask
            denominator = mask.sum() + self.epsilon
        else:
            denominator = torch.numel(pred) + self.epsilon
        
        # Mean similarity is analogous to F1 score for regression
        f1 = similarity.sum() / denominator
        
        # Return 1 - f1 as the loss (0 = perfect match)
        return 1.0 - f1


class SIP2NetLoss(nn.Module):
    """
    SIP2Net loss using library implementations where available
    """
    
    def __init__(self, alpha1=500, alpha2=1, alpha3=1, use_mse=True, mse_weight=1.0):
        super().__init__()
        
        # Use PIQ for SSIM (higher quality implementation)
        self.ssim_loss = piq.SSIMLoss(data_range=255.0)
        
        # GDL is not commonly available, so use custom implementation
        self.gdl_loss = CustomGradientDifferenceLoss(alpha=1)
        
        # Use a custom F1 loss that works better for regression
        self.f1_loss = CustomF1Loss()
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.use_mse = use_mse
        self.mse_weight = mse_weight
    
    def forward(self, pred, target, mask=None, weights=None):
        # Ensure inputs have channel dimension
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if weights is not None and weights.dim() == 3:
            weights = weights.unsqueeze(1)
        
        # Calculate SSIM loss
        # PIQ expects normalized inputs
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
        if mask is not None:
            # Apply mask before calling
            pred_norm = pred_norm * mask
            target_norm = target_norm * mask
        
        ssim = self.ssim_loss(pred_norm, target_norm)
        
        # Calculate GDL loss
        gdl = self.gdl_loss(pred, target, mask)
        
        # Calculate F1 loss (using our custom implementation)
        f1 = self.f1_loss(pred, target, mask)
        
        # Combine losses
        sip2net_loss = self.alpha1 * ssim + self.alpha2 * gdl + self.alpha3 * f1
        
        # Add MSE if requested
        if self.use_mse and mask is not None:
            sq_error = (pred - target) ** 2 * mask
            if weights is not None:
                sq_error *= weights
            mse = sq_error.sum() / (mask.sum() + 1e-8)
            total_loss = sip2net_loss + self.mse_weight * mse
            mse_value = mse.item()
        else:
            total_loss = sip2net_loss
            mse_value = 0.0
        
        # Return total loss and components
        components = {
            'ssim_loss': ssim.item(),
            'gdl_loss': gdl.item(),
            'f1_loss': f1.item(),
            'sip2net_loss': sip2net_loss.item(),
            'mse': mse_value,
            'total_loss': total_loss.item()
        }
        
        return total_loss, components


def create_sip2net_loss(use_mse=True, mse_weight=0.5, alpha1=500, alpha2=1, alpha3=1):
    """
    Create a SIP2Net loss instance with specified parameters
    
    Args:
        use_mse: Whether to include MSE in the total loss
        mse_weight: Weight of MSE component relative to SIP2Net losses
        alpha1: Weight of SSIM loss
        alpha2: Weight of Gradient Difference loss
        alpha3: Weight of F1 loss
        
    Returns:
        SIP2NetLoss instance
    """
    return SIP2NetLoss(
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        use_mse=use_mse,
        mse_weight=mse_weight
    )
