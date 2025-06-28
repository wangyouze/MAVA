import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.append('/data/home/wangyouze/projects/Adversarial-Training-against-Multimodal-Adversarial-Attacks/adversarial_robustness_VE/')
from models.model_ve import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from transformers import BertForMaskedLM
import utils
sys.path.append('/data/home/wangyouze/projects/Adversarial-Training-against-Multimodal-Adversarial-Attacks/adversarial_robustness/')
from torchattacks.attacks.autoattack import AutoAttack
from torchvision import transforms
from dataset import ve_dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import copy

def evaluate(model, model_ad, ref_model, data_loader, tokenizer, device, config, args):
    # test
    model.eval()
    ref_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 1

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    attack = AutoAttack(model, norm='Linf', n_classes=3, eps=args.eps/255, version='rand', seed=0, verbose=False)

    for images, texts, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        images, targets = images.to(device), targets.to(device)

        text_inputs = tokenizer(texts, padding='longest', return_tensors="pt").to(device)
        if args.adv != 0:

            images = attack(images, texts, targets, tokenizer)

        images = images_normalize(images)
        

        with torch.no_grad():
            prediction = model_ad(images, text_inputs, targets=targets, train=False)

        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

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

def main(args, config):
    # device = args.gpu[0]
    device='cuda'

    # fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    model = load_model(tokenizer, args, device)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])
    datasets = ve_dataset(config['test_file'], test_transform, config['image_root'])
    test_loader = DataLoader(datasets, batch_size=args.batch_size, num_workers=4)

    model_ad = copy.deepcopy(model)
    if args.fine_tuning:
        visual_encoder_state_dict = torch.load(args.fine_tuning_checkpoint, map_location='cuda')
        model_ad.visual_encoder.load_state_dict(visual_encoder_state_dict, strict=False)


    print("Start eval")
    test_stats = evaluate(model, model_ad, ref_model, test_loader, tokenizer, device, config, args)

    print(test_stats)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/data/home/wangyouze/projects/Adversarial-Training-against-Multimodal-Adversarial-Attacks/adversarial_robustness_VE/configs/VE.yaml')
    parser.add_argument('--output_dir', default='output_fusion/VE')
    # parser.add_argument('--checkpoint', default="/data/home/wangyouze/MultimodalAttack/Co-attack/checkpoints/VE/ALBEF-VE.pth")
    # parser.add_argument('--fine_tuning_checkpoint', default="/data/home/wangyouze/projects/save/adversarial_robustness/VE/ALBEF/VE_ALBEF_TeCoA_eps_2_lr_1e_4/ViT-B_16_openai_imagenet_l2_imagenet_XxVZA/checkpoints/step_9.pt")
    parser.add_argument('--checkpoint', default="/data/home/wangyouze/MultimodalAttack/Co-attack/checkpoints/VE/TCL-VE.pth")
    # parser.add_argument('--fine_tuning_checkpoint', default="/data/home/wangyouze/projects/save/adversarial_robustness/VE/TCL/TVT_TCL_lr_0.001_eps_2.0_stepsize_1.0_N_5_pgd_2/ViT-B_16_openai_train_coco_2_test_flickr30k_l2_train_coco_2_test_flickr30k_POZnf/checkpoints/step_9.pt")
    parser.add_argument('--fine_tuning_checkpoint', default="/data/home/wangyouze/projects/save/adversarial_robustness/TCL/VE/VE_TCL_TeCoA_eps_2_lr_1e_4/ViT-B_16_openai_imagenet_l2_imagenet_dYBTS/checkpoints/step_9.pt")
    parser.add_argument('--text_encoder', default="/data/home/wangyouze/projects/checkpoints/bert-base-uncased/")
    parser.add_argument('--gpu', type=int, nargs='+',  default=[0])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--adv', default=1, type=int)
    parser.add_argument('--cls', action='store_true')
    parser.add_argument('--alpha', default=3.0, type=float)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--eps', default=2.0, type=float)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # if args.epsilon:
    #     config['epsilon'] = args.epsilon

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)

