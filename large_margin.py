import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def _max_with_relu(a, b):
    return a + F.relu(b - a)

    
def _get_grad(out_, in_):
    grad, *_ = torch.autograd.grad(out_, in_,
                                   grad_outputs=torch.ones_like(out_, dtype=torch.float32),
                                   retain_graph=True)
    return grad.view(in_.shape[0], -1)

class LargeMarginLoss:
    """Large Margin Loss
    A Pytorch Implementation of `Large Margin Deep Networks for Classification`
    Referenced to Official TF Repo ( https://github.com/google-research/google-research/tree/master/large_margin )
    Docs is written with referenced to Official TF Repo
    
    Arguments : 
          gamma (float): Desired margin, and distance to boundary above the margin will be clipped.
          alpha_factor (float): Factor to determine the lower bound of margin.
              Both gamma and alpha_factor determine points to include in training 
              the margin these points lie with distance to boundary of [gamma * (1 - alpha), gamma]
          top_k (int):Number of top classes to include in the margin loss.
          dist_norm (1, 2, np.inf): Distance to boundary defined on norm
          epslion (float): Small number to avoid division by 0.
          use_approximation (bool):
          loss_type ("all_top_k", "worst_top_k", "avg_top_k"):  If 'worst_top_k'
              only consider the minimum distance to boundary of the top_k classes. If
              'average_top_k' consider average distance to boundary. If 'all_top_k'
              consider all top_k. When top_k = 1, these choices are equivalent.
    """
    def __init__(self, 
                 gamma=10000.0,
                 alpha_factor=4.0,
                 top_k=1,
                 dist_norm=2,
                 epsilon=1e-8,
                 use_approximation=True,
                 loss_type="all_top_k"):
        
        self.dist_upper = gamma
        self.dist_lower = gamma * (1.0 - alpha_factor)
        
        self.alpha = alpha_factor
        self.top_k = top_k
        self.dual_norm = {1: np.inf, 2: 2, np.inf: 1}[dist_norm]
        self.eps = epsilon
        
        self.use_approximation = use_approximation
        self.loss_type = loss_type

    def __call__(self, logits, onehot_labels, feature_maps):
        """Getting Large Margin loss
        
        Arguments : 
            logits (Tensor): output of Network before softmax
            onehot_labels (Tensor): One-hot shaped label
            feature_maps (list of Tensor): Target feature maps(Layer of NN) want to enforcing by Large Margin
            
        Returns :
            loss:  Large Margin loss
        """
        prob = F.softmax(logits, dim=1)
        correct_prob = prob * onehot_labels

        correct_prob = torch.sum(correct_prob, dim=1, keepdim=True)
        other_prob = prob * (1.0 - onehot_labels)
        
        if self.top_k > 1:
            topk_prob, _ = other_prob.topk(self.top_k, dim=1)
        else:
            topk_prob, _ = other_prob.max(dim=1, keepdim=True)
        
        diff_prob = correct_prob - topk_prob
        
        loss = torch.empty(0, device=logits.device)
        for feature_map in feature_maps:
            diff_grad = torch.stack([_get_grad(diff_prob[:, i], feature_map) for i in range(self.top_k)],
                                    dim=1)
            diff_gradnorm = torch.norm(diff_grad, p=self.dual_norm, dim=2)

            if self.use_approximation:
                diff_gradnorm.detach_()
                
            dist_to_boundary = diff_prob / (diff_gradnorm + self.eps)
            
            if self.loss_type == "worst_top_k":
                dist_to_boundary, _ = dist_to_boundary.min(dim=1)
            elif self.loss_type == "avg_top_k":
                dist_to_boundary = dist_to_boundary.mean(dim=1)
                        
            loss_layer = _max_with_relu(dist_to_boundary, self.dist_lower)
            loss_layer = _max_with_relu(0, self.dist_upper - loss_layer) - self.dist_upper
            loss = torch.cat([loss, loss_layer])
        return loss.mean()
    