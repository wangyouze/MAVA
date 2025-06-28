import torch
import torch.nn.functional as F
from enum import Enum

class NormType(Enum):
    Linf = 0
    L2 = 1

def clamp_by_l2(x, max_norm):
    norm = torch.norm(x, dim=(1,2,3), p=2, keepdim=True)
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    return x * factor

def random_init(x, norm_type, epsilon):
    delta = torch.zeros_like(x)
    if norm_type == NormType.Linf:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data * epsilon
    elif norm_type == NormType.L2:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data - x
        delta.data = clamp_by_l2(delta.data, epsilon)
    return delta

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

class ImageAttacker():
    # APGD (Auto-PGD)
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, cls=False, 
                 rho=0.75, alpha_init=None, *args, **kwargs):
        self.norm_type = norm_type
        self.random_init = random_init
        self.epsilon = epsilon
        self.cls = cls
        self.preprocess = kwargs.get('preprocess')
        self.bounding = kwargs.get('bounding')
        if self.bounding is None:
            self.bounding = (0, 1)
        
        # APGD specific parameters
        self.rho = rho  # step size reduction factor
        self.alpha_init = alpha_init if alpha_init is not None else epsilon / 4
        self.checkpoints = [0.22, 0.5, 0.75]  # checkpoints for step size reduction
        
    def input_diversity(self, image):
        return image
    
    def attack(self, image, num_iters):
        batch_size = image.shape[0]
        
        # Initialize perturbation
        if self.random_init:
            self.delta = random_init(image, self.norm_type, self.epsilon)
        else:
            self.delta = torch.zeros_like(image)
            
        if hasattr(self, 'kernel'):
            self.kernel = self.kernel.to(image.device)
        if hasattr(self, 'grad'):
            self.grad = torch.zeros_like(image)
        
        # APGD parameters
        self.alpha = self.alpha_init
        self.best_loss = float('-inf') * torch.ones(batch_size, device=image.device)
        self.best_delta = self.delta.clone()
        self.reduced_last_check = torch.zeros(batch_size, dtype=torch.bool, device=image.device)
        self.n_reduced = torch.zeros(batch_size, dtype=torch.long, device=image.device)
        
        # Momentum for APGD
        self.momentum = torch.zeros_like(image)
        self.mu = 0.9  # momentum coefficient
        
        checkpoint_iters = [int(p * num_iters) for p in self.checkpoints]
        
        for i in range(num_iters):
            self.delta = self.delta.detach()
            self.delta.requires_grad = True
            
            image_diversity = self.input_diversity(image + self.delta)
            
            if self.preprocess is not None:
                image_diversity = self.preprocess(image_diversity)
            
            yield image_diversity
            
            grad = self.get_grad()
            
            # Update momentum
            self.momentum = self.mu * self.momentum + grad
            
            # Normalize gradient
            grad_normalized = self.normalize(self.momentum)
            
            # Update delta
            self.delta = self.delta.data + self.alpha * grad_normalized
            
            # Project to epsilon ball
            self.delta = self.project(self.delta, self.epsilon)
            
            # Project to valid image range
            self.delta = torch.clamp(image + self.delta, *self.bounding) - image
            
            # APGD step size adaptation
            if i in checkpoint_iters:
                self.adapt_step_size(i, num_iters)
        
        # Return best perturbation found
        yield (image + self.best_delta).detach()
    
    def adapt_step_size(self, current_iter, total_iters):
        """Adapt step size based on progress"""
        # Check if we should reduce step size
        checkpoint_idx = None
        for idx, checkpoint_iter in enumerate([int(p * total_iters) for p in self.checkpoints]):
            if current_iter == checkpoint_iter:
                checkpoint_idx = idx
                break
        
        if checkpoint_idx is not None:
            # Reduce step size if not much progress
            mask_reduce = ~self.reduced_last_check
            self.alpha = torch.where(mask_reduce, self.alpha * self.rho, self.alpha)
            self.n_reduced += mask_reduce.long()
            self.reduced_last_check = torch.zeros_like(self.reduced_last_check)
    
    def get_grad(self):
        self.grad = self.delta.grad.clone()
        return self.grad
    
    def project(self, delta, epsilon):
        if self.norm_type == NormType.Linf:
            return torch.clamp(delta, -epsilon, epsilon)
        elif self.norm_type == NormType.L2:
            return clamp_by_l2(delta, epsilon)
    
    def normalize(self, grad):
        if self.norm_type == NormType.Linf:
            return torch.sign(grad)
        elif self.norm_type == NormType.L2:
            return grad / torch.norm(grad, dim=(1, 2, 3), p=2, keepdim=True)
    
    def run_trades(self, net, image, num_iters):
        batch_size = image.shape[0]
        
        with torch.no_grad():
            origin_output = net.inference_image(self.preprocess(image))
            if self.cls:
                origin_embed = origin_output['image_embed'][:, 0, :].detach()
            else:
                origin_embed = origin_output['image_embed'].flatten(1).detach()
        
        criterion = torch.nn.KLDivLoss(reduction='none')  # 改为'none'以便逐样本处理
        attacker = self.attack(image, num_iters)
        
        # Track best adversarial examples
        self.best_loss = float('-inf') * torch.ones(batch_size, device=image.device)
        self.best_delta = torch.zeros_like(image)
        
        for i in range(num_iters):
            image_adv = next(attacker)
            adv_output = net.inference_image(image_adv)
            
            if self.cls:
                adv_embed = adv_output['image_embed'][:, 0, :]
            else:
                adv_embed = adv_output['image_embed'].flatten(1)
            
            # Calculate loss for each sample
            loss_per_sample = criterion(adv_embed.log_softmax(dim=-1), 
                                       origin_embed.softmax(dim=-1)).sum(dim=-1)
            loss = loss_per_sample.mean()  # Mean for backward pass
            
            # Update best adversarial examples
            mask_better = loss_per_sample > self.best_loss
            self.best_loss = torch.where(mask_better, loss_per_sample, self.best_loss)
            
            # Update best delta for samples that improved
            for j in range(batch_size):
                if mask_better[j]:
                    self.best_delta[j] = self.delta[j].clone()
            
            loss.backward()
            
            # Check for step size reduction at checkpoints
            checkpoint_iters = [int(p * num_iters) for p in self.checkpoints]
            if i in checkpoint_iters:
                # Mark samples that haven't improved much for step size reduction
                if hasattr(self, 'prev_best_loss'):
                    improvement = self.best_loss - self.prev_best_loss
                    self.reduced_last_check = improvement < 0.1 * self.best_loss.abs()
                self.prev_best_loss = self.best_loss.clone()
        
        image_adv = next(attacker)
        return image_adv
    
    def run_apgd_targeted(self, net, image, target_embed, num_iters):
        """APGD for targeted attacks"""
        batch_size = image.shape[0]
        criterion = torch.nn.MSELoss(reduction='none')
        attacker = self.attack(image, num_iters)
        
        # Track best adversarial examples (minimize distance to target)
        self.best_loss = float('inf') * torch.ones(batch_size, device=image.device)
        self.best_delta = torch.zeros_like(image)
        
        for i in range(num_iters):
            image_adv = next(attacker)
            adv_output = net.inference_image(image_adv)
            
            if self.cls:
                adv_embed = adv_output['image_embed'][:, 0, :]
            else:
                adv_embed = adv_output['image_embed'].flatten(1)
            
            # Calculate distance to target for each sample
            loss_per_sample = criterion(adv_embed, target_embed).sum(dim=-1)
            loss = loss_per_sample.mean()
            
            # Update best adversarial examples (lower loss is better for targeted)
            mask_better = loss_per_sample < self.best_loss
            self.best_loss = torch.where(mask_better, loss_per_sample, self.best_loss)
            
            for j in range(batch_size):
                if mask_better[j]:
                    self.best_delta[j] = self.delta[j].clone()
            
            # For targeted attack, we want to minimize the loss
            (-loss).backward()  # Negative because we want to minimize distance
            
            # Step size adaptation
            checkpoint_iters = [int(p * num_iters) for p in self.checkpoints]
            if i in checkpoint_iters:
                if hasattr(self, 'prev_best_loss'):
                    improvement = self.prev_best_loss - self.best_loss  # Improvement is reduction in loss
                    self.reduced_last_check = improvement < 0.1 * self.best_loss
                self.prev_best_loss = self.best_loss.clone()
        
        image_adv = next(attacker)
        return image_adv