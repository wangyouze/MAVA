import torch
import sys
sys.path.append('/data/home/wangyouze/projects/Adversarial-Training-against-Multimodal-Adversarial-Attacks/adversarial_robustness/vlm_eval')
from attacks.utils import project_perturbation, normalize_grad

from torchvision import transforms
import numpy as np

def pgd(
        model,
        loss_fn,
        data_clean,
        norm,
        eps,
        iterations,
        stepsize,
        perturbation=None,
        mode='min',
        momentum=0.9,
        verbose=False
):
    """
    Minimize or maximize given loss
    """
    # make sure data is in image space
    assert torch.max(data_clean) < 1. + 1e-6 and torch.min(data_clean) > -1e-6

    if perturbation is None:
        perturbation = torch.zeros_like(data_clean, requires_grad=True)
    velocity = torch.zeros_like(data_clean)
    for i in range(iterations):
        perturbation.requires_grad = True
        with torch.enable_grad():
            out = model.encode_image(data_clean + perturbation)
            out_norm = out / out.norm(dim=-1, keepdim=True)   
            loss = loss_fn(out_norm)
            if verbose:
                print(f'[{i}] {loss.item():.5f}')

        with torch.no_grad():
            gradient = torch.autograd.grad(loss, perturbation)[0]
            if gradient.isnan().any():  #
                print(f'attention: nan in gradient ({gradient.isnan().sum()})')  #
                gradient[gradient.isnan()] = 0.
            # normalize
            gradient = normalize_grad(gradient, p=norm)
            # momentum
            velocity = momentum * velocity + gradient
            velocity = normalize_grad(velocity, p=norm)
            # update
            if mode == 'min':
                perturbation = perturbation - stepsize * velocity
            elif mode == 'max':
                perturbation = perturbation + stepsize * velocity
            else:
                raise ValueError(f'Unknown mode: {mode}')
            # project
            perturbation = project_perturbation(perturbation, eps, norm)
            perturbation = torch.clamp(
                data_clean + perturbation, 0, 1
            ) - data_clean  # clamp to image space
            assert not perturbation.isnan().any()
            assert torch.max(data_clean + perturbation) < 1. + 1e-6 and torch.min(
                data_clean + perturbation
            ) > -1e-6

            # assert (ctorch.compute_norm(perturbation, p=self.norm) <= self.eps + 1e-6).all()
    # todo return best perturbation
    # problem is that model currently does not output expanded loss
    return data_clean + perturbation.detach()

def vnifgsm(
        txt2img,
        model,
        loss_fn,
        images,
        eps,
        iterations,
        stepsize,
        decay,
        norm=False,
        momentum=0.9,
        N = 5,
        beta=3/2,
        device='cuda',
):
    images = images.clone().detach().to(device)

    momentum = torch.zeros_like(images).detach().to(device)
    v = torch.zeros_like(images).detach().to(device)

    adv_images = images.clone().detach()

    for _ in range(iterations):
        adv_images.requires_grad = True
        nes_images = adv_images + decay * stepsize * momentum
        out = model.inference_image(nes_images)['image_feat']
        if norm:
            out = out / out.norm(dim=-1, keepdim=True)
        cost = loss_fn(out, txt2img)
       

        # Update adversarial images
        adv_grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]

        grad = (adv_grad + v) / torch.mean(
            torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
        )
        grad = grad + momentum * decay
        momentum = grad

        # Calculate Gradient Variance
        GV_grad = torch.zeros_like(images).detach().to(device)
        for _ in range(N):
            neighbor_images = adv_images.detach() + torch.randn_like(
                images
            ).uniform_(-eps * beta, eps * beta)
            neighbor_images.requires_grad = True
            outputs = model.inference_image(neighbor_images)['image_feat']
            if norm:
                outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            

            cost = loss_fn(outputs, txt2img)
            GV_grad += torch.autograd.grad(
                cost, neighbor_images, retain_graph=False, create_graph=False
            )[0]

        # obtaining the gradient variance
        v = GV_grad / N - adv_grad

        adv_images = adv_images.detach() + stepsize * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    neighbor_images_set = []
    
    for _ in range(N):
        neighbor_images = adv_images.detach() + torch.randn_like(
            images
        ).uniform_(-eps, eps)
        neighbor_images_set.append(neighbor_images)
    adv_images_neighbor = torch.cat(neighbor_images_set, dim=0)
    return adv_images_neighbor, adv_images



def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        
        reverse_transform = transforms.Resize(ori_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio*ori_shape[0]), 
                                  int(ratio*ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            
            reversed_imgs = reverse_transform(scaled_imgs)
            
            result.append(reversed_imgs)
        
        return torch.cat([imgs,]+result, 0)


def attacker(
        txt2img,
        model,
        loss_fn,
        images,
        eps,
        iterations,
        stepsize,
        decay,
        scales,
        momentum=0.9,
        N = 5,
        beta=3/2,
        device='cuda',
):
    images = images.clone().detach().to(device)

    momentum = torch.zeros_like(images).detach().to(device)
    v = torch.zeros_like(images).detach().to(device)

    adv_images = images.clone().detach()

    ori_shape = (images.shape[-2], images.shape[-1])
        
    reverse_transform = transforms.Resize(ori_shape,
                            interpolation=transforms.InterpolationMode.BICUBIC)

    for _ in range(iterations):
        adv_images.requires_grad = True
        nes_images = adv_images + decay * stepsize * momentum
        out = model.encode_image(nes_images)
        out_norm = out / out.norm(dim=-1, keepdim=True)
        cost = loss_fn(out_norm, txt2img)
       

        # Update adversarial images
        adv_grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]

        grad = (adv_grad + v) / torch.mean(
            torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
        )
        grad = grad + momentum * decay
        momentum = grad

        # Calculate Gradient Variance
        GV_grad = torch.zeros_like(images).detach().to(device)
        for _ in range(N):
            neighbor_images = adv_images.detach() + torch.randn_like(
                images
            ).uniform_(-eps * beta, eps * beta)
            
            neighbor_images.requires_grad = True
            outputs = model.encode_image(neighbor_images)
            outputs_norm = outputs / outputs.norm(dim=-1, keepdim=True)

            cost = loss_fn(outputs_norm, txt2img)
            GV_grad += torch.autograd.grad(
                cost, neighbor_images, retain_graph=False, create_graph=False
            )[0]

        # for ratio in scales:
        #     scale_shape = (int(ratio*ori_shape[0]), 
        #                         int(ratio*ori_shape[1]))
        #     scale_transform = transforms.Resize(scale_shape,
        #                         interpolation=transforms.InterpolationMode.BICUBIC)
        #     scaled_imgs = adv_images.detach() + torch.from_numpy(np.random.normal(0.0, 0.05, adv_images.shape)).uniform_(-eps * beta, eps * beta).to(device)
        #     scaled_imgs = scale_transform(scaled_imgs)
            
        #     reversed_imgs = reverse_transform(scaled_imgs)
        #     reversed_imgs.requires_grad = True
        #     outputs = model.encode_image(reversed_imgs)
        #     outputs_norm = outputs / outputs.norm(dim=-1, keepdim=True)

        #     cost = loss_fn(outputs_norm, txt2img)
        #     GV_grad += torch.autograd.grad(
        #         cost, reversed_imgs, retain_graph=False, create_graph=False
        #     )[0]




        # obtaining the gradient variance
        v = GV_grad / N - adv_grad

        adv_images = adv_images.detach() + stepsize * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        neighbor_images_set = []
    
        for _ in range(N):
            neighbor_images = adv_images.detach() + torch.from_numpy(np.random.normal(0.0, 0.05, adv_images.shape)).uniform_(-eps * beta, eps * beta).to(device)
            neighbor_images_set.append(neighbor_images)
        adv_images_neighbor = torch.cat(neighbor_images_set, dim=0)

    return adv_images_neighbor, adv_images

def pgd_2(
        txt2img,
        model,
        loss_fn,
        images,
        eps,
        iterations,
        stepsize,
        decay,
        momentum=0.9,
        N = 5,
        beta=3/2,
        device='cuda',
):
    images = images.clone().detach().to(device)

    momentum = torch.zeros_like(images).detach().to(device)
    v = torch.zeros_like(images).detach().to(device)

    adv_images = images.clone().detach()

    neighbor_images_set = []

    for _ in range(iterations):
        adv_images.requires_grad = True
        nes_images = adv_images + decay * stepsize * momentum
        
        out = model.inference_image(nes_images)['image_feat']
        out_norm = out / out.norm(dim=-1, keepdim=True)
        cost = loss_fn(out_norm, txt2img)
       

        # Update adversarial images
        adv_grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]

        grad = (adv_grad + v) / torch.mean(
            torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
        )
        grad = grad + momentum * decay
        momentum = grad

        adv_images = adv_images.detach() + stepsize * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        neighbor_images_set.append(adv_images)
    
    adv_images_h = torch.zeros_like(adv_images)
    for im in neighbor_images_set[iterations - N - 1:]:
        adv_images_h += im
    # adv_images_h += adv_images
    adv_images_h /= (N+1)
    # adv_images_h = torch.cat(neighbor_images_set[:3], dim=0)
    print('adv_images_h_shape=', adv_images_h.shape)
   
    return adv_images_h, adv_images