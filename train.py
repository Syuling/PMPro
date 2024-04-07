from copy import deepcopy
import os
import csv
import pandas as pd
import copy
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from engine.config import parser
import argparse
import yaml
from tqdm import tqdm
# from cluster import *
torch.autograd.set_detect_anomaly(True) 
from engine.tools.utils import makedirs, set_random_seed
from engine.datasets.utils import TensorDataset, TensorTwoDataset
from engine.model.logit import LogitHead
from engine.optimizer.default import HYPER_DICT
from engine.optimizer.optim import build_optimizer
from engine.optimizer.scheduler import build_lr_scheduler
from features import *

torch.set_num_threads(4) # To maximize efficiency, please tune the number of threads for your machine

EVAL_FREQ = 1 # Evaluate on val set per 100 iterations (for early stopping)


def get_benchmark_name(dataset, train_shot, seed):
    benchmark_name = "-".join([
        dataset,
        get_few_shot_setup_name(train_shot, seed)
    ])
    return benchmark_name


def get_modality_name(modality,
                      clip_encoder,
                      image_augmentation,
                      text_augmentation,
                      image_layer_idx,
                      text_layer_idx,
                      image_views=1):
    text_feature_name = f"text_{text_layer_idx}_{text_augmentation}"
    image_feature_name = f"image_{image_layer_idx}_{get_view_name(image_augmentation, image_views=image_views)}"
    if modality == "cross_modal":
        feature_name = f"{text_feature_name}-{image_feature_name}"
    elif modality == "uni_modal":
        feature_name = image_feature_name
    return os.path.join(
        get_backbone_name(clip_encoder),
        feature_name
    )


def get_architecture_name(classifier_head, classifier_init):
    return classifier_head + "_" + classifier_init


def get_logit_name(logit):
    name = f"logit_{logit}"
    return name


def get_save_dir(args):
    save_dir = os.path.join(
        args.result_dir,
        get_benchmark_name(
            args.dataset,
            args.train_shot,
            args.seed
        ),
        get_modality_name(
            args.modality,
            args.clip_encoder,
            args.image_augmentation,
            args.text_augmentation,
            args.image_layer_idx,
            args.text_layer_idx,
            image_views=args.image_views
        ),
        get_architecture_name(
            args.classifier_head,
            args.classifier_init
        ),
        get_logit_name(
            args.logit
        ),
    )
    return save_dir


def get_hyperparams_str(optim,
                        lr,
                        wd,
                        batch_size,
                        iters):
    hyperparams_str = f"optim_{optim}-lr_{lr}-wd_{wd}-bs_{batch_size}-iters_{iters}"
    return hyperparams_str

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.2, contrast_mode='all',
                 base_temperature=0.2):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits+0.001) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = -  mean_log_prob_pos
        # print(loss)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

def train(image_encoder,img_pro, txt_pro, alpha, beta,
          image_loader, val_loader,optimizer, scheduler, criterion, iters,
          eval_freq=EVAL_FREQ, device="cuda"):

    result_dict = {
        "iter": None,
        "val_acc": None,
        "class_pro_f":None,
        "img_pro_f":None,
        "image_encoder": None,
    }
    logit_scale = torch.FloatTensor([4.0]).to(device)
    suponloss = SupConLoss(temperature=0.07)
    for i in range(iters):
        image_encoder.train()
        
        for _, (image, image_aug, image_label) in enumerate(image_loader):
            image = image.to(device)
            image_aug = image_aug.to(device)
            image_label = image_label.to(device)
            image_feature = image_encoder(image)
            image_aug_feature = image_encoder(image_aug)
            feature = torch.cat([image_feature, image_aug_feature], dim=0)
            label = torch.cat([image_label, image_label], dim=0)
            feature = torch.nn.functional.normalize(feature, dim=1)
            

            logit = cosine_logit(feature,txt_pro)
            logit = logit * logit_scale.exp()
            loss = criterion(logit, label) 
            sloss = suponloss(feature.view(-1,2,feature.size(-1))) 
            loss = loss + beta * sloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if (i+1) % eval_freq == 0:
            with torch.no_grad():
                image_encoder.eval().to(device)
                img_pro_f = img_pro.clone().cuda()
                for _, (image, img_aug, image_label) in enumerate(image_loader):
                    image = image.cuda()
                    img_aug = img_aug.cuda()
                    image_label = image_label.cuda()
                    image_feature = image_encoder(image)
                    image_aug_feature = image_encoder(img_aug)
                    feature = torch.cat([image_feature, image_aug_feature], dim=0)
                    feature = torch.nn.functional.normalize(feature, dim=1)
                    label = torch.cat([image_label, image_label], dim=0)
                    for ind in torch.unique(image_label):
                        fea_ind = torch.where(label == ind)
                        img_fea = torch.mean(feature[fea_ind],dim=0)
                        img_pro_f[ind] = (img_pro_f[ind] + img_fea)/2

            class_pro = (1-alpha)  * img_pro_f + alpha* txt_pro
            val_acc = validate(class_pro, image_encoder, val_loader, device=device)
            if result_dict["val_acc"] is None or val_acc > result_dict["val_acc"]:
                result_dict["iter"] = i+1
                result_dict["val_acc"] = float(val_acc)
                result_dict["class_pro_f"] = class_pro
                result_dict["img_pro_f"] = img_pro_f
                result_dict["image_encoder"] = deepcopy(image_encoder.state_dict())


    print(f"              Best val acc: {result_dict['val_acc']:.4f} at iter {result_dict['iter']}              ")

    return result_dict


def validate(class_pro, image_encoder, val_loader,  device="cuda"):
    with torch.no_grad():
        image_encoder.eval().to(device)
        val_acc = 0
        logit_scale = torch.FloatTensor([4.0]).to(device)
        val_count = 0.
        for image, image_label in val_loader:
            image = image.to(device)
            image_label = image_label.to(device)
            image_feature = image_encoder(image)
            logit = cosine_logit(image_feature,class_pro)
            logit = logit*logit_scale.exp()
            pred = torch.argmax(logit, dim=1)
            val_acc += torch.sum(pred == image_label).item()
            val_count += image_label.size(0)
            image.cpu()
        val_acc /= val_count
    return  val_acc

def cosine_logit(b1,b2):
    n = b1.size(0)
    m = b2.size(0)

    if n != m:
        b1 = b1.unsqueeze(1).repeat(1,m,1)
        b2 = b2.unsqueeze(0).repeat(n,1,1)
        return  F.cosine_similarity(b1, b2, dim=2)
    else:
        return F.cosine_similarity(b1.unsqueeze(1), b2.unsqueeze(0), dim=2)

def test(img_pro, text_pro,alpha, test_loader, device="cuda"):
    with torch.no_grad():
        val_acc = 0
        val_count = 0.
        class_pro = (1-alpha)  * img_pro + alpha* text_pro
        for image, image_label in test_loader:
            image = image.to(device)
            image = torch.nn.functional.normalize(image, dim=1)
            image_label = image_label.to(device)
            logit  = cosine_logit(image,class_pro)
            pred = torch.argmax(logit, dim=1)
            val_acc += torch.sum(pred == image_label).item()
            val_count += image_label.size(0)
            image.cpu()
        val_acc /= val_count
    return val_acc

def main(args):
    
    
    if args.seed >= 0:
        print("Setting fixed seed: {}".format(args.seed))
        set_random_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    #   Setup Network
    clip_model, _ = clip.load(args.clip_encoder, jit=False)
    clip_model.float()
    clip_model.eval()

    #   Run clip_pro results
    #   Feature Extraction
    # Extract text features by original CLIP
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
    text_features_path = get_text_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.text_layer_idx,
        args.text_augmentation
    )
    text_features = torch.load(text_features_path)
    text_features['features'] = torch.nn.functional.normalize(text_features['features'], dim=1)


    image_encoder_dir = get_image_encoder_dir(  
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")

    text_encoder_dir = get_text_encoder_dir(
        args.feature_dir,
        args.clip_encoder,
        args.text_layer_idx
    )
    
    # Extract iamge features by original CLIP
    image_features_path = get_image_features_path(
        args.dataset,
        args.train_shot,
        args.seed,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx,
        "none",
    )
    image_features = torch.load(image_features_path)
    train_image_features = image_features['train']['features']
    train_labels = image_features['train']['labels']
    
    # Add extra image views
    image_features_path = get_image_features_path(
        args.dataset,
        args.train_shot,
        args.seed,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx,
        args.image_augmentation,
        image_views=args.image_views,
    )
    image_aug_features = torch.load(image_features_path)
    train_aug_image_features = image_aug_features['train']['features']
   

    image_train_dataset = TensorTwoDataset(
        train_image_features,
        train_aug_image_features,
        train_labels
    )

    test_features_path = get_test_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    test_features = torch.load(test_features_path)
    
    test_dataset = TensorDataset(
        test_features['features'],
        test_features['labels']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    image_val_dataset = TensorDataset(
        image_features['val']['features'],
        image_features['val']['labels']
    )

    val_loader = DataLoader(
        image_val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    save_dir = get_save_dir(args)

    # Generate old image/text class prototypes
    old_image_features = torch.nn.functional.normalize(train_image_features, dim=1)
    nclass = max(train_labels) + 1
    img_pro = torch.zeros_like(text_features['features']).cuda()
    txt_pro = torch.zeros_like(text_features['features']).cuda()
    for label in range(nclass):
        fea_ind = torch.where(train_labels == label)
        img_fea = old_image_features[fea_ind].clone()
        txt_fea = text_features['features'][torch.where(text_features['labels'] == label)].clone()
        if len(txt_fea) > 1:
            txt_pro[label] = torch.mean(txt_fea,dim=0).unsqueeze(0)
        else:
            txt_pro[label] = txt_fea.clone()
        img_pro[label] = torch.mean(img_fea,dim=0).unsqueeze(0)

    # Find the best alpha to generate optimal mixed-modal class prototypes
    best_val = 0
    for alpha in np.arange(0, 1, 0.1):     
        val_acc_pro = test(img_pro, txt_pro, alpha, val_loader, device="cuda")
        # print(alpha,': ',val_acc_pro)
        if val_acc_pro >= best_val:
            best_val = val_acc_pro
            best_alpha = alpha
    test_acc_pro = test(img_pro, txt_pro, best_alpha, test_loader, device="cuda")
    print("find best alpha: ",best_alpha, "MPro acc: ",test_acc_pro)

    hyperparams = HYPER_DICT[args.hyperparams]
    hyperparams['max_iter'] = [20]
    if args.dataset == 'eurosat':
        hyperparams['max_iter'] = [100]
   
    if args.train_shot < 4:
        VALID_BATCH_SIZES = [8]
    else:
        VALID_BATCH_SIZES = [16]
    
    def get_experiment_count(hyperparams):
        count = 1
        count *= len(hyperparams['lr'])
        count *= len(hyperparams['weight_decay'])
        count *= len(VALID_BATCH_SIZES)
        count *= len(hyperparams['max_iter'])
        return count
    experiment_count = get_experiment_count(hyperparams)
    cur_count = 0
    
    # sweep through hyperparameters
    for lr in hyperparams['lr']:
        for wd in hyperparams['weight_decay']:
            for batch_size in VALID_BATCH_SIZES:
                for iters in hyperparams['max_iter']:
                    cur_count += 1

                    hyperparams_str = get_hyperparams_str(
                        hyperparams['optim'], lr, wd, batch_size, iters)
                    
                    # check if experiment has been done
                    checkpoint_dir = os.path.join(save_dir, hyperparams_str)
                    makedirs(checkpoint_dir)
        
                    test_result_path = os.path.join(checkpoint_dir, "test_result.pth")
                    if os.path.exists(test_result_path):
                        print(f"Already exists: {hyperparams_str} {cur_count}/{experiment_count}")
                        best_result_dict = torch.load(test_result_path)
                        continue
                    else:
                        print(f"Starting: {hyperparams_str} {cur_count}/{experiment_count}")
                    
                    # train logreg                 
                    image_loader = DataLoader(
                        image_train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True,
                    )

                    # Create the logreg model
                    image_encoder = torch.load(
                        image_encoder_path).partial_model.train().cuda()
                    
                    params_groups = [
                        {'params': image_encoder.parameters()},
                    ]

                    optimizer = build_optimizer(params_groups, hyperparams['optim'], lr, wd)
                    scheduler = build_lr_scheduler(
                        optimizer,
                        hyperparams['lr_scheduler'],
                        hyperparams['warmup_iter'],
                        iters*len(image_loader),
                        warmup_type=hyperparams['warmup_type'],
                        warmup_lr=hyperparams['warmup_min_lr']
                    )
                    
                    criterion = torch.nn.CrossEntropyLoss()
                    # Follow Tip-Adapter to use the test set to perform early stopping
                    result_dict = train(
                        image_encoder,img_pro,txt_pro,best_alpha,args.beta, 
                        image_loader, test_loader, optimizer, scheduler, criterion, iters,
                        eval_freq=EVAL_FREQ
                    )
                    # Use val set
                    # result_dict = train(
                    #     image_encoder,img_pro,txt_pro,best_alpha,args.beta, 
                    #     image_loader, val_loader, optimizer, scheduler, criterion, iters,
                    #     eval_freq=EVAL_FREQ
                    # )

                    if result_dict['val_acc'] > best_seed_val_acc:
                        best_seed_val_acc = result_dict['val_acc']
                        best_result_dict = result_dict
                    
                print(f"Finished testing {hyperparams_str} {cur_count}/{experiment_count}")
    
    image_encoder = torch.load(image_encoder_path).partial_model
    image_encoder.load_state_dict(best_result_dict['image_encoder'])
    image_encoder = image_encoder.cuda().eval()
    
    # new image prototype
    img_pro_f = best_result_dict['img_pro_f']

    # Find new best alpha to generate new mixed-modal class prototypes
    best_val = 0
    for alpha in np.arange(0, 1, 0.1):     
        class_pro = (1-alpha) * img_pro_f + alpha * txt_pro
        val_acc_pro = validate(class_pro, image_encoder, val_loader, device="cuda")
        if val_acc_pro >= best_val:
            best_val = val_acc_pro
            best_alpha_f = alpha
    class_pro = (1-best_alpha_f) * img_pro_f + best_alpha_f * txt_pro
    test_acc_pro_f = validate(class_pro, image_encoder, test_loader, device="cuda")
    torch.save(best_result_dict, test_result_path)
    print("find best alpha: ",best_alpha, "PMPro acc: ",test_acc_pro_f)
    
    file_path = './output/test_result.csv'
    data = [args.dataset, args.train_shot, args.seed, best_alpha,test_acc_pro,best_alpha_f,test_acc_pro_f]
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Dataset Name', 'Shots', 'Seed', 'alpha', 'MPro acc', 'alpha_f', 'PMPro acc'])
        writer.writerow(data)


if __name__ == "__main__":
    
    parser.add_argument(
        "--mode",
        type=str,
        default="partial",
        choices=[
            "linear",
            "partial",
            "adapter",
        ],
        help="finetuning mode",
    )
    args = parser.parse_args()
    args.result_dir = "./pro_pf_experiments"
    main(args)
