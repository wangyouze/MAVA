import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import shutil
import time
import string
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.scheduler import cosine_lr
from torchvision import transforms
import sys
sys.path.append('/data/home/wangyouze/projects/github/MAVA/adversarial_robustness_VE/')
from models.tokenization_bert import BertTokenizer
from models.model_ve import ALBEF
from models.vit import interpolate_pos_embed
from dataset import ve_dataset, pair_dataset

from transformers import BertForMaskedLM
from torchvision import transforms
from PIL import Image

import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default='openai')
parser.add_argument('--dataset', type=str, default='train_coco_2_test_flickr30k')
parser.add_argument('--optimizer_state', type=str, default='', help='Optimizer state file path')
parser.add_argument('--steps', type=int, default=10, help='Number of training steps')
parser.add_argument('--warmup', type=int, default=14000, help='Warmup steps')
parser.add_argument('--train_batch_size', type=int, default=7)
parser.add_argument('--loss', type=str, default='l2', help='ce, l2')
parser.add_argument('--opt', type=str, default='sgd', help='Optimizer type; sgd, adamw')
parser.add_argument('--momentum_sgd', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--attack', type=str, default='pgd', help='Adversarial attack type')
parser.add_argument('--inner_loss', type=str, default='l2', help='Inner loss function for adversarial training')
parser.add_argument('--norm', type=str, default='linf', help='Norm for adversarial perturbation')
parser.add_argument('--eps', type=float, default=4., help='Epsilon for adversarial perturbation')
parser.add_argument('--iterations_adv', type=int, default=10, help='Iterations for adversarial attack')
parser.add_argument('--stepsize_adv', type=float, default=1.0, help='Step size for adversarial attack (no effect for apgd)')
parser.add_argument('--overwrite', type=bool, default=False, help='Overwrite existing directory')
parser.add_argument('--log_freq', type=int, default=1, help='Logging frequency')
parser.add_argument('--eval_freq', type=int, default=50, help='Evaluation frequency')
parser.add_argument('--save_checkpoints', type=bool, default=True, help='Save 10 training checkpoints')
parser.add_argument('--devices', type=str, default='', help='Device IDs for CUDA')

parser.add_argument('--num_neighbor_sampling', type=int, default=5, help='num_neighbor_sampling')
parser.add_argument('--num_image_captions', type=int, default=5, help='num_neighbor_sampling')

parser.add_argument('--model', default='ViT-B/16', type=str)
parser.add_argument('--text_encoder', default="/data/home/wangyouze/MultimodalAttack/Co-attack/checkpoints/bert-base-uncased", type=str)
parser.add_argument('--config', default='/data/home/wangyouze/projects/Adversarial-Training-against-Multimodal-Adversarial-Attacks/adversarial_robustness_VG/configs/Retrieval_coco.yaml')
parser.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5')
parser.add_argument('--adversarial_attack', type=bool, default=True)
parser.add_argument('--checkpoint', default="/data/home/wangyouze/MultimodalAttack/Co-attack/checkpoints/VE/ALBEF-VE.pth")
parser.add_argument("--local-rank", default=-1, type=int)
parser.add_argument("--seed", default=2014, type=int)



def load_model(tokenizer, args, device):
	

	model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

	### load checkpoint
	checkpoint = torch.load(args.checkpoint, map_location='cpu')

	try:
		state_dict = checkpoint['model']
	except:
		state_dict = checkpoint

	for key in list(state_dict.keys()):
		if 'bert' in key:
			encoder_key = key.replace('bert.', '')
			state_dict[encoder_key] = state_dict[key]
			del state_dict[key]
	msg = model.load_state_dict(state_dict, strict=False)

	print('load checkpoint from %s' % args.checkpoint)
	# print(msg)

	model = model.to(device)


	return model


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
        
        out = model.inference_image(nes_images)['image_embed'][:, 0, :]
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
    # print('adv_images_h_shape=', adv_images_h.shape)
   
    return adv_images_h, adv_images

def main(args, config):
	local_rank = args.local_rank
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend='nccl')
	
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
		

	# print args
	print(f"Arguments:\n{'-' * 20}")
	for arg, value in vars(args).items():
		print(f"{arg}: {value}")
	print(f"{'-' * 20}")

	# setup dirs
	if args.overwrite:
		shutil.rmtree(args.output_dir, ignore_errors=True)
	os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=False)

	# write args to file
	with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
		f.write(str(args))

	
	 #### Model ####
	
	tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
	model_orig = load_model(tokenizer, args, device='cuda')
	model_orig = DDP(model_orig, device_ids=[local_rank], output_device=local_rank)
	
	model = load_model(tokenizer, args, device='cuda')
	model = DDP(model, device_ids=[local_rank], output_device=local_rank)
	#print(msg)
	

	# get data
	train_transform = transforms.Compose([
		transforms.Resize((384, 384), interpolation=Image.BICUBIC),
		transforms.ToTensor(),
	])

	# train_dataset = ve_dataset(config['test_file'], train_transform, config['image_root'])

	train_dataset = pair_dataset(config['test_file'], train_transform, config['image_root'], mode='test')

	train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, 
						   num_workers=4,
						   collate_fn=train_dataset.collate_fn,
						  sampler=train_sampler
						  )

	# set optimizer (all params have requires_grad=True)

	for key, param in model.module.named_parameters():
		if 'visual_encoder' in key:
			# print(key)
			param.requires_grad = True
		else:
			param.requires_grad = False
	params = model.module.visual_encoder.parameters()


	if args.opt == 'adamw':
		# optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
		optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
	elif args.opt == 'sgd':
		optimizer = torch.optim.SGD(
			params,
			lr=args.lr,
			momentum=args.momentum_sgd,
			weight_decay=args.wd
		)
	else:
		raise ValueError(f'Optimizer {args.optimizer} not supported.')
	if args.optimizer_state != '':
		optimizer.load_state_dict(torch.load(args.optimizer_state))

	# set scheduler
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  T_max=100, eta_min=1e-4, verbose=True)


	# compute amount of epochs
	total_epochs = args.steps / len(train_loader)
	print(f'train for {total_epochs} epochs')
	args.total_epochs = total_epochs

	# finetune
	loss_buffer = []
	
	for epoch in tqdm(range(args.steps)):
		train_loader.sampler.set_epoch(epoch)
		loss_total = train_one_epoch(
			model=model.module,
			model_orig=model_orig.module,
			dataloader=train_loader,
			optimizer=optimizer,
			scheduler=scheduler,
			args=args,
			tokenizer=tokenizer,
		)
		print(f'Epoch {epoch} done.')
		if local_rank == 0:
			loss_buffer.append(loss_total)
		
			print('loss=',loss_total)
		
		
			if args.save_checkpoints:
				# save model and optimizer state_dict
				torch.save(model.module.visual_encoder.state_dict(), f'{args.output_dir}/checkpoints/step_{epoch}.pt')
				torch.save(optimizer.state_dict(), f'{args.output_dir}/checkpoints/step_{epoch}_opt.pt')

			# plot_loss(loss_buffer, args=args)




class ComputeLossWrapper:
	def __init__(self, ori_text_embedding, ori_image_embedding,  reduction='mean', loss=None,
				 logit_scale=100.):
		self.ori_text_embedding = ori_text_embedding
		self.ori_image_embedding = ori_image_embedding
		self.reduction = reduction
		self.loss_str = loss
		self.logit_scale = logit_scale

	def __call__(self, adv_image_embedding, txt2img):
		return compute_loss(ori_image_embedding=self.ori_image_embedding,
							adv_image_embedding=adv_image_embedding,
							ori_text_embedding=self.ori_text_embedding ,
							txt2img=txt2img,
							)
		
def train_one_epoch(
		model, model_orig, dataloader, optimizer, scheduler,
		args, tokenizer, device='cuda',
):
	model_orig.eval()
	model.train()

	if args.scales is not None:
		scales = [float(itm) for itm in args.scales.split(',')]
		print(scales)

	N = args.num_neighbor_sampling
	M = args.num_image_captions

	loss_buffer = []
	
	for batch_idx, (images, texts_group, images_ids, text_ids_groups) in enumerate(dataloader):

		print(f'--------------------> batch:{batch_idx}/{len(dataloader)}')
		texts_ids = []
		txt2img = []
		texts = []
		for i in range(len(texts_group)):
			if len(texts_group[i]) >= M:
				texts += texts_group[i][:M]
				texts_ids += text_ids_groups[i][:M]
				txt2img += [i]*len(text_ids_groups[i][:M])
			else:
				texts += texts_group[i]
				texts_ids += text_ids_groups[i]
				txt2img += [i]*len(text_ids_groups[i])


		images = images.to(device).repeat_interleave(M, dim=0)       
							
		with torch.no_grad():
			texts_input = tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
			output = model_orig.inference(images, texts_input, use_embeds=False)
			image_features = output['image_embed'][:, 0, :]
			text_features = output['text_embed'][:, 0, :]
	

		clean_image_features = model.inference_image(images)['image_embed'][:, 0, :]

		# loss for the attack
		loss_inner_wrapper = ComputeLossWrapper(
			ori_text_embedding=text_features,
			ori_image_embedding=image_features,
			reduction='none' if args.attack == 'apgd' else 'mean', loss=args.inner_loss,
			logit_scale=100.
			)
		model.eval()

		adv_images_neighbor, adv_images = pgd_2(
				txt2img=txt2img,
				model=model,
				loss_fn=loss_inner_wrapper,
				images=images,
				eps=args.eps,
				iterations=10,
				stepsize=args.stepsize_adv,
				decay=1.0,
				momentum=0.9,
				N = N,
				
				device='cuda',
		)
	

		del loss_inner_wrapper
		model.train()

		adv_image_features_neighbor = model.inference_image(adv_images_neighbor)['image_embed'][:, 0, :]

		# image_features_ = image_features.repeat(N, 1)
		loss_1 = l2(adv_image_features_neighbor, image_features, reduction='mean')
		print('loss_1=', loss_1.item())

		loss_2 = loss_multi_view(adv_image_features_neighbor, text_features, txt2img)
		# loss_2 = l2(adv_image_features, text_features, reduction='mean')
		
		loss_3 = l2(clean_image_features, image_features, reduction='mean')



		loss_total = 0.4 * loss_1 + 0.8* loss_3 + 0.002 * loss_2
		loss_buffer.append(loss_total.item())
		print('loss_total=', loss_total.item())

		
		optimizer.zero_grad()
		loss_total.backward()
		optimizer.step()
		# scheduler.step()
	   


		torch.cuda.empty_cache()
	
	return sum(loss_buffer)/len(loss_buffer), 0, 0, 0


def plot_loss(loss_buffer, args):

	sns.set_theme()
	num_iters = len(loss_buffer)

	x_ticks = list(range(0, num_iters))

	# Plot and label the training and validation loss values
	plt.plot(x_ticks, loss_buffer, label='Target Loss')

	# Add in a title and axes labels
	plt.title('Loss Plot')
	plt.xlabel('Iters')
	plt.ylabel('Loss')

	# Display the plot
	plt.legend(loc='best')
	plt.savefig('%s/loss_curve.png' % (args.save_dir))
	plt.clf()

	# torch.save(loss_buffer, '%s/loss.txt' % (args.save_dir))
	with open('%s/loss.json' % (args.save_dir), 'w') as file:
		json.dump(loss_buffer, file)

def loss_multi_view(adv_imgs_embeds, txts_embeds, txt2img):  
	device = adv_imgs_embeds.device    

	it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
	it_labels = torch.zeros(it_sim_matrix.shape).to(device)
	
	for i in range(len(txt2img)):
		it_labels[txt2img[i], i]=1
	
	loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
	loss = loss_IaTcpos
	
	return loss


def compute_loss(ori_image_embedding, adv_image_embedding, ori_text_embedding, txt2img):

	loss_1 = l2(out=adv_image_embedding, targets=ori_image_embedding, reduction='mean')

	loss_3 = loss_multi_view(adv_image_embedding, ori_text_embedding, txt2img)
		
	# adv_image_embedding = adv_image_embedding.repeat(5, 1)
	loss_2 = l2(out=adv_image_embedding, targets=ori_text_embedding, reduction='mean')
	return loss_1 + loss_2 + loss_3

def l2(out, targets, reduction='none'):
	# squared l2 - it does not divide by the latent dimension
	# should have shape (batch_size, embedding_size)
	assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
	assert out.shape[0] > 1
	# Compute the element-wise squared error
	squared_error_batch = F.mse_loss(out, targets, reduction='none')
	if reduction == 'mean':
		squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
	else:
		squared_error_batch = squared_error_batch.sum(dim=1)
		assert squared_error_batch.shape == (out.shape[0],), f'{squared_error_batch.shape} != {(out.shape[0],)}'
	return squared_error_batch

def ce(out, targets, reduction='mean'):
	# out = logits
	assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
	assert out.shape[0] > 1

	return F.cross_entropy(out, targets, reduction=reduction)

if __name__ == '__main__':
	# set seeds
	torch.manual_seed(0)
	np.random.seed(0)

	# Parse command-line arguments
	args = parser.parse_args()

	args.output_dir = f'./save/ALBEF/MAVA_lr_{args.lr}_eps_{args.eps}_stepsize_{args.stepsize_adv}_N_{args.num_neighbor_sampling}_beta_0.8'
	# args.save_dir = f'/data/home/wangyouze/projects/save/adversarial_robustness/VE/ALBEF/TVT_ALBEF_lr_{args.lr}_eps_{args.eps}_stepsize_{args.stepsize_adv}_N_{args.num_neighbor_sampling}_beta_0.8'
	args.eps /= 255
	args.stepsize_adv /= 255
	# make sure there is no string in args that should be a bool
	assert not any([isinstance(x, str) and x in ['True', 'False'] for x in args.__dict__.values()]), f'args contains a string that should be a bool: {args}'
	assert args.eval_freq % args.log_freq == 0, 'eval_freq must be a multiple of log_freq'

	if args.devices != '':
		# set cuda visible devices
		os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

	random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
	args.finetuned_model_name = f'{args.model}_{args.pretrained}_{args.dataset}_{args.loss}_{args.dataset}_{random_str}'
	args.finetuned_model_name = args.finetuned_model_name.replace('/', '_')
	args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)

	# run

	config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
	main(args, config)