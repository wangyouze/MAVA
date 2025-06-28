import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption


class ve_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.labels = {'entailment':2,'neutral':1,'contradiction':0}
        self.text = []
        for i in range(len(self.ann)):
            sentence = pre_caption(self.ann[i]['sentence'], self.max_words)
            self.text.append(sentence)

        self.positive_ann = []
        for j, ann in enumerate(self.ann):
            # if j not in targets_ids:
            #     continue
            if self.labels[ann['label']] == 2:
                self.positive_ann.append(ann)
                # print(ann['image'])
                     

    def __len__(self):
        return len(self.positive_ann)
    

    def __getitem__(self, index):    
        
        ann = self.positive_ann[index]
        
        image_path = os.path.join(self.image_root+'flickr30k-images','%s.jpg'%ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(ann['sentence'], self.max_words)

        return image, sentence, self.labels[ann['label']], ann['image']

import torch
class ve_dataset_SGA(Dataset):
    def __init__(self, ann_file, flikcr_ann_file, transform, image_root, max_words=30):        
        
        self.ann = json.load(open(ann_file, 'r'))
        self.flickr_ann = json.load(open(flikcr_ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.labels = {'entailment':2,'neutral':1,'contradiction':0}

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        self.positive_ann = []
        for i, ann in enumerate(self.ann):
            if self.labels[ann['label']] == 2:
                self.positive_ann.append(ann)

        self.flickr_img2txt = {}
        for i, ann in enumerate(self.flickr_ann):
            img_id = ann['image'].split('/')[-1].split('.')[0]
            texts_ = []
            for j, caption in enumerate(ann['caption']):
                texts_.append(pre_caption(caption, self.max_words))
            self.flickr_img2txt[img_id] = texts_

        for i, ann_ in enumerate(self.positive_ann):
            self.img2txt[i] = []
            self.image.append(ann_['image'])
            sentence = pre_caption(ann_['sentence'], self.max_words)
            self.text.append(sentence)

            self.txt2img[txt_id] = i
            self.img2txt[i].append(txt_id)
            txt_id += 1
            if ann_['image'] in self.flickr_img2txt.keys():
                for j, cap in enumerate(self.flickr_img2txt[ann_['image']][:4]):
                    self.text.append(cap)
                    self.txt2img[txt_id] = i
                    self.img2txt[i].append(txt_id)
                    txt_id += 1
                
            else:
                for _ in range(4):
                    self.text.append(ann_['sentence'])
                    self.txt2img[txt_id] = i
                    self.img2txt[i].append(txt_id)
                    txt_id += 1
               
      
    def __len__(self):
        return len(self.positive_ann)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root+'flickr30k-images', self.image[index]+'.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        text_ids =  self.img2txt[index]
        texts = [self.text[i] for i in self.img2txt[index]]
        return image, texts, index, text_ids

    def collate_fn(self, batch):
        imgs, txt_groups, img_ids, text_ids_groups = list(zip(*batch))        
        imgs = torch.stack(imgs, 0)
        return imgs, txt_groups, list(img_ids), text_ids_groups
    
    
class ve_dataset_SGA_2(Dataset):
    def __init__(self, ann_file, text_aug_path, transform, image_root, max_words=30):        
        
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text_aug_list = []
        with open(text_aug_path, 'r', encoding='utf-8') as f_aug:
            for line in f_aug:
                new_line = line.strip().split('\t')
                new_line = new_line[1:]
                self.text_aug_list.append(new_line)


        self.labels = {'entailment':2,'neutral':1,'contradiction':0}

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        self.positive_ann = []
        for i, ann in enumerate(self.ann):
            if self.labels[ann['label']] == 2:
                self.positive_ann.append(ann)


        for i, ann_ in enumerate(self.positive_ann):
            self.img2txt[i] = []
            self.image.append(ann_['image'])
            sentence = pre_caption(ann_['sentence'], self.max_words)
            self.text.append(sentence)

            self.txt2img[txt_id] = i
            self.img2txt[i].append(txt_id)
            txt_id += 1
        
            for j in range(4):
                self.text.append(self.text_aug_list[i][j])
                self.txt2img[txt_id] = i
                self.img2txt[i].append(txt_id)
                txt_id += 1
               
      
    def __len__(self):
        return len(self.positive_ann)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root+'flickr30k-images', self.image[index]+'.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        text_ids =  self.img2txt[index]
        texts = [self.text[i] for i in self.img2txt[index]]
        return image, texts, index, text_ids

    def collate_fn(self, batch):
        imgs, txt_groups, img_ids, text_ids_groups = list(zip(*batch))        
        imgs = torch.stack(imgs, 0)
        return imgs, txt_groups, list(img_ids), text_ids_groups
    
    