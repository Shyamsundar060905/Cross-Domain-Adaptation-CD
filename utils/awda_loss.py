import torch
import torch.nn as nn
import torch.nn.functional as F

class AWDA_Manager:
    def __init__(self,device,num_classes=2, alpha=0.90):
        self.p_glb = torch.zeros(num_classes, device=device)
        self.alpha = alpha

    def update_weights(self, pred_s, y_s, curr_iter, total_iter):
        with torch.no_grad():
            prob = F.softmax(pred_s, dim=1)
            for c in range(2):
                mask = (y_s == c)
                if mask.any():
                    p_cur = prob[:, c][mask].mean()
                    self.p_glb[c] = self.alpha * self.p_glb[c] + (1 - self.alpha) * p_cur
        
        exponent = 2 * (1 - curr_iter / total_iter) + 1 # 3 -> 1
        return 1.0 / (self.p_glb ** exponent)

    def get_cwst_loss(self, pred_t, pseudo_label, weights, curr_iter, total_iter):
        threshold = 0.93

        
        prob_t = F.softmax(pred_t, dim=1)
        max_prob, _ = prob_t.max(dim=1)
    
        conf_mask = (max_prob > threshold).float()
    
        # pixel-wise cross entropy
        loss = F.cross_entropy(pred_t, pseudo_label, reduction='none')
    
        # apply weights and confidence mask
        loss = loss * weights[pseudo_label] * conf_mask
        return loss.sum() / (conf_mask.sum() + 1e-6)
    def get_pixelwise_loss(self, pred_t, pseudo_label, weights):
        # pixel-wise CE
        loss = F.cross_entropy(pred_t, pseudo_label, reduction='none')
        
        # apply class-wise weights
        loss = loss * weights[pseudo_label]
        
        return loss