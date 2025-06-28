import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from transformers import BertForMaskedLM

from models.model_ve import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import ve_dataset
from PIL import Image
from torchvision import transforms
from attack import *

def evaluate(model, model_ad, ref_model, data_loader, tokenizer, device, config, args):
    # test
    model.eval()
    ref_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    image_attacker = ImageAttacker(args.eps/ 255., preprocess=images_normalize, bounding=(0, 1), cls=args.cls)
    text_attacker = BertAttackFusion(ref_model, tokenizer, cls=args.cls)
    multi_attacker = MultiModalAttacker(model, image_attacker, text_attacker, tokenizer, cls=args.cls)

    for images, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        images, targets = images.to(device), targets.to(device)
        if args.adv != 0:
            images, text = multi_attacker.run(images, text, adv=args.adv, num_iters=config['num_iters'],
                                              alpha=args.alpha)

        images = images_normalize(images)
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

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
    device = args.gpu[0]

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])
    datasets = ve_dataset(config['test_file'], test_transform, config['image_root'])
    test_loader = DataLoader(datasets, batch_size=args.batch_size, num_workers=4)

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = load_model(tokenizer, args, device)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)

    model_ad = load_model(tokenizer, args, device)

    # visual_encoder_state_dict = torch.load("/data1/wangyouze/projects/save/VE/TCL/TVT_lr_0.001_eps_2/ViT-B_16_openai_train_coco_2_test_flickr30k_l2_train_coco_2_test_flickr30k_Fsr1p/checkpoints/step_10.pt", map_location='cuda')

    # visual_encoder_state_dict = torch.load("/data1/wangyouze/projects/save/VE/ALBEF/TVT_lr_0.001_eps_2/ViT-B_16_openai_train_coco_2_test_flickr30k_l2_train_coco_2_test_flickr30k_p1ojW/checkpoints/step_10.pt", map_location='cuda')

    # visual_encoder_state_dict = torch.load( "/data1/wangyouze/projects/save/VE/TCL/TVT_lr_0.001_eps_2/ViT-B_16_openai_train_coco_2_test_flickr30k_l2_train_coco_2_test_flickr30k_Fsr1p/checkpoints/step_10.pt", map_location='cuda')


    # visual_encoder_state_dict = torch.load( "/data1/wangyouze/projects/save/VE/TCL/TVT_TCL_lr_0.001_eps_2.0_stepsize_1.0_N_3_pgd_2/ViT-B_16_openai_train_coco_2_test_flickr30k_l2_train_coco_2_test_flickr30k_czHTq/checkpoints/step_10.pt", map_location='cuda')
    # visual_encoder_state_dict = torch.load( "/data1/wangyouze/projects/save/VE/ALBEF/TVT_ALBEF_lr_0.001_eps_2.0_stepsize_1.0_N_3_pgd_2/ViT-B_16_openai_train_coco_2_test_flickr30k_l2_train_coco_2_test_flickr30k_rztap/checkpoints/step_10.pt", map_location='cuda')
   
    # visual_encoder_state_dict = torch.load( "/data1/wangyouze/projects/save/VE/TCL/TVT_TCL_lr_0.001_eps_4.0_stepsize_1.0_N_3_pgd_2/ViT-B_16_openai_train_coco_2_test_flickr30k_l2_train_coco_2_test_flickr30k_z3Bt3/checkpoints/step_10.pt", map_location='cuda')
    # visual_encoder_state_dict = torch.load( "/data1/wangyouze/projects/save/VE/ALBEF/TVT_ALBEF_lr_0.001_eps_4.0_stepsize_1.0_N_3_pgd_2/ViT-B_16_openai_train_coco_2_test_flickr30k_l2_train_coco_2_test_flickr30k_G7W3M/checkpoints/step_10.pt", map_location='cuda')
    # visual_encoder_state_dict = torch.load("/data1/wangyouze/projects/save/VE/ALBEF/TVT_ALBEF_lr_0.0001_eps_4.0_stepsize_1.0_N_3_beta_0.8_pgd_2/ViT-B_16_openai_train_coco_2_test_flickr30k_l2_train_coco_2_test_flickr30k_cGAN4/checkpoints/step_10.pt", map_location='cuda')
    visual_encoder_state_dict = torch.load("/data1/wangyouze/projects/save/VE/TCL/TVT_TCL_lr_0.001_eps_2.0_stepsize_1.0_N_10_pgd_2/ViT-B_16_openai_train_coco_2_test_flickr30k_l2_train_coco_2_test_flickr30k_kULa8/checkpoints/step_10.pt", map_location='cuda')
    
    model_ad.visual_encoder.load_state_dict(visual_encoder_state_dict, strict=False)

    model = model.to(device)
    model_ad = model_ad.to(device)
    ref_model = ref_model.to(device)


    print("Start evaluating")
    start_time = time.time()

    test_stats = evaluate(model, model_ad, ref_model, test_loader, tokenizer, device, config, args)

    print(test_stats)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluating time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/data/wangyouze/projects/adversarial_robustness_VE/configs/VE.yaml')
    parser.add_argument('--output_dir', default='output_fusion/VE')
    # parser.add_argument('--checkpoint', default="/data/wangyouze/projects/adversarial_robustness_VE/VE/ALBEF-VE.pth")
    parser.add_argument('--checkpoint', default="/data/wangyouze/projects/adversarial_robustness_VE/VE/TCL-VE.pth")
    parser.add_argument('--text_encoder', default='/data/wangyouze/checkpoints/bert-base-uncased/')
    parser.add_argument('--gpu', type=int, nargs='+',  default=[0])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--adv', default=0, type=int)
    parser.add_argument('--cls', action='store_true')
    parser.add_argument('--alpha', default=3.0, type=float)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--eps', default=2.0, type=float)
    parser.add_argument('--batch_size', default=80, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # if args.epsilon:
    #     config['epsilon'] = args.epsilon

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)