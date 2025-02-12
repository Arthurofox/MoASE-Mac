# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Task-specific Losses
# -------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class JsdCrossEntropy(nn.Module):
    """
    Jensen-Shannon Divergence + Cross-Entropy Loss.
    
    This loss splits the batch into several parts, computes cross-entropy on the clean part,
    and adds an averaged KL divergence term across splits.
    """
    def __init__(self, num_splits=3, alpha=12, smoothing=0.1):
        super(JsdCrossEntropy, self).__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        if smoothing is not None and smoothing > 0:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        split_size = output.shape[0] // self.num_splits
        if split_size == 0:
            # If the batch size is smaller than num_splits, fall back to basic cross-entropy
            return self.cross_entropy_loss(output, target)
        
        # Adjust split size to ensure it fits within the batch size
        if split_size * self.num_splits != output.shape[0]:
            split_size = output.shape[0] // self.num_splits
            split_size = max(1, split_size)  # Ensure split size is at least 1

        logits_split = torch.split(output, split_size)

        # Compute cross-entropy loss on the clean (first) split
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]
        
        # Compute the mixture distribution and clamp values to avoid extreme KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        kl_losses = [F.kl_div(logp_mixture, p_split, reduction='batchmean') for p_split in probs]
        loss += self.alpha * sum(kl_losses) / len(kl_losses)
        return loss

# -------------------------------
# Additional Losses
# -------------------------------

class ConsistencyLoss(nn.Module):
    """
    Consistency Loss encourages the student model's predictions to be similar to the teacher's.
    Here, we use a simple MSE loss between soft logits.
    """
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor) -> torch.Tensor:
        return self.mse_loss(student_outputs, teacher_outputs)


class HomeostaticProximalLoss(nn.Module):
    """
    Homeostatic-Proximal (HP) Loss.
    
    This loss is composed of a task-specific loss (e.g., cross-entropy) plus a regularization term
    that penalizes the deviation of student expert parameters from the teacher's.
    
    The overall loss is:
        L_total = L_task(student_outputs, targets) + (mu/2) * Σ||θ_S_e - θ_T_e||²
    where the summation is over expert modules.
    """
    def __init__(self, base_loss_fn: nn.Module = None, mu: float = 0.1):
        super(HomeostaticProximalLoss, self).__init__()
        self.base_loss_fn = base_loss_fn if base_loss_fn is not None else nn.CrossEntropyLoss()
        self.mu = mu

    def forward(self, student_outputs: torch.Tensor, targets: torch.Tensor,
                student_expert_params: list, teacher_expert_params: list) -> torch.Tensor:
        base_loss = self.base_loss_fn(student_outputs, targets)
        reg_loss = 0.0
        for s_param, t_param in zip(student_expert_params, teacher_expert_params):
            reg_loss += torch.norm(s_param - t_param, p=2) ** 2
        hp_loss = 0.5 * self.mu * reg_loss
        return base_loss + hp_loss

# -------------------------------
# Composite Loss Module
# -------------------------------

class CompositeLoss(nn.Module):
    """
    Composite Loss for Continual Test-Time Adaptation.
    
    This composite loss combines:
      1. A task-specific loss (JSD Cross-Entropy or Label Smoothing Cross-Entropy)
      2. A Homeostatic-Proximal loss to regularize expert parameters.
      3. (Optional) A consistency loss to align teacher and student predictions.
    
    The overall loss is given by:
        L_total = L_task + λ_hp * L_HP + λ_cons * L_consistency
    """
    def __init__(self, 
                 task_loss_type: str = 'jsd', 
                 num_splits: int = 3, 
                 alpha: float = 12, 
                 smoothing: float = 0.1, 
                 lambda_hp: float = 0.1, 
                 lambda_cons: float = 0.0):
        super(CompositeLoss, self).__init__()
        self.lambda_hp = lambda_hp
        self.lambda_cons = lambda_cons
        
        if task_loss_type == 'jsd':
            self.task_loss_fn = JsdCrossEntropy(num_splits=num_splits, alpha=alpha, smoothing=smoothing)
        elif task_loss_type == 'label_smoothing':
            self.task_loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            self.task_loss_fn = nn.CrossEntropyLoss()
        
        # Homeostatic-Proximal loss uses the same task loss as base.
        self.hp_loss_fn = HomeostaticProximalLoss(base_loss_fn=self.task_loss_fn, mu=lambda_hp)
        
        if lambda_cons > 0:
            self.consistency_loss_fn = ConsistencyLoss()
        else:
            self.consistency_loss_fn = None

    def forward(self, student_outputs: torch.Tensor, targets: torch.Tensor,
                student_expert_params: list, teacher_expert_params: list,
                teacher_outputs: torch.Tensor = None) -> torch.Tensor:
        # Compute the task-specific loss (or base loss)
        task_loss = self.task_loss_fn(student_outputs, targets)
        # Compute the homeostatic-proximal regularization loss
        hp_loss = self.hp_loss_fn(student_outputs, targets, student_expert_params, teacher_expert_params)
        
        total_loss = task_loss + hp_loss
        
        if self.consistency_loss_fn is not None and teacher_outputs is not None:
            cons_loss = self.consistency_loss_fn(student_outputs, teacher_outputs)
            total_loss = total_loss + self.lambda_cons * cons_loss
        
        return total_loss

# -------------------------------
# For Testing Purposes
# -------------------------------
if __name__ == "__main__":
    # Dummy data for testing
    batch_size, num_classes = 12, 10
    student_outputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Simulate expert parameters (using two dummy tensors per expert)
    student_expert_params = [torch.randn(5, 5), torch.randn(3, 3)]
    teacher_expert_params = [p + 0.1 * torch.randn_like(p) for p in student_expert_params]
    
    # Optionally, simulate teacher outputs for consistency loss
    teacher_outputs = torch.randn(batch_size, num_classes)
    
    # Create composite loss instance with JSD and a small consistency weight
    comp_loss_fn = CompositeLoss(task_loss_type='jsd', num_splits=3, alpha=12, smoothing=0.1,
                                   lambda_hp=0.1, lambda_cons=0.05)
    loss_value = comp_loss_fn(student_outputs, targets, student_expert_params, teacher_expert_params, teacher_outputs)
    print("Composite Loss:", loss_value.item())
