from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import sys
import yaml
import time
import h5py
import logging
import argparse
import numpy as np
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict
import torch
import torch.nn as nn
import imageio
from pretrain_provider import Provider
from utils.seed import setup_seed
import torch.nn.functional as F
from utils.visual2d import visual_2d
from mim.vit_3d import ViT
from mim.vit_orgmim import MAE
from timm.utils import ModelEma


def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M',
            filename=path,
            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if not os.path.exists(cfg.cache_path):
        os.makedirs(cfg.cache_path)
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    if not os.path.exists(cfg.record_path):
        os.makedirs(cfg.record_path)
    if not os.path.exists(cfg.valid_path):
        os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer


def load_dataset(cfg):
    print('Caching datasets ... ', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider


def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters,
                                                                  cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(
                1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0

    model = ViT(
        image_size=cfg.MODEL.img_size, 
        frames=cfg.MODEL.img_size,  
        image_patch_size=cfg.MODEL.patch_size,  
        frame_patch_size=cfg.MODEL.patch_size, 
        channels=1,
        num_classes=1000,
        dim=cfg.MODEL.dim,
        depth=cfg.MODEL.depth,
        heads=cfg.MODEL.head,
        mlp_dim=cfg.MODEL.mlp_dim,
        dropout=0.1,
        emb_dropout=0.1
    )

    if cfg.MODEL.continue_train:
        ckpt_path = cfg.MODEL.continue_path
        print('Load pre-trained model from' + ckpt_path)
        checkpoint = torch.load(ckpt_path + '/model.ckpt')
        new_state_dict = OrderedDict()
        state_dict = checkpoint['model_weights']
        for k, v in state_dict.items():
            name = k.replace('module.', '') if 'module' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    learner = MAE(
        encoder=model,
        masking_ratio=cfg.TRAIN.mask_ratio,
        decoder_dim=512,  
        decoder_depth=6,  
    )

    if cfg.MODEL.continue_train:
        ckpt_path = cfg.MODEL.continue_path
        print('Load pre-trained model from' + ckpt_path)
        checkpoint = torch.load(ckpt_path + '/learner.ckpt')
        new_state_dict = OrderedDict()
        state_dict = checkpoint['model_weights']
        for k, v in state_dict.items():
            name = k.replace('module.', '') if 'module' in k else k
            new_state_dict[name] = v
        learner.load_state_dict(new_state_dict)

    learner_ema = ModelEma(learner, decay=0.999, device='cuda:0', resume='')
    learner_ema.decay = 0.999
    optimizer = torch.optim.AdamW(learner.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999))

    learner = learner.cuda()

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            learner = nn.DataParallel(learner)
        else:
            raise AttributeError(
                'Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)

    N = cfg.TRAIN.loss_forward
    while iters <= cfg.TRAIN.total_iters:
        learner.train()
        iters += 1
        t1 = time.time()
        inp, att = train_provider.next()
        inp, att = inp.cuda(), att.cuda()
        optimizer.zero_grad()
        ######### Loss Branch #########
        patch_num = (cfg.MODEL.img_size // cfg.MODEL.patch_size) ** 3
        num_masked = int(learner.masking_ratio * patch_num)
        rand_indices = torch.rand(cfg.TRAIN.batch_size, patch_num, device='cuda:0').argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        batch_range = torch.arange(cfg.TRAIN.batch_size, device='cuda:0')[:, None]
        for i in range(N):
            with torch.no_grad():
                pred_pixel_values, masked_patches = learner_ema.ema(inp, masked_indices, unmasked_indices, num_masked)
                recon_loss_m = ((pred_pixel_values - masked_patches) ** 2).mean(dim=2, keepdim=False)  
                recon_loss = torch.zeros([cfg.TRAIN.batch_size, patch_num], device='cuda:0')
                recon_loss[batch_range, masked_indices] = recon_loss_m

        masked_indices_l, unmasked_indices_l = learner.generate_mask_alm(recon_loss, step=iters, total_step=cfg.TRAIN.total_iters, alpha_t=cfg.TRAIN.alpha_t)
        pred_pixel_values_l, masked_patches_l = learner(inp, masked_indices_l, unmasked_indices_l, num_masked)
        loss1 = F.mse_loss(pred_pixel_values_l, masked_patches_l)

        ######### Membrane Branch #########
        masked_indices_s, unmasked_indices_s = learner.generate_mask_mam(att, step=iters, total_step=cfg.TRAIN.total_iters, patch_size=cfg.MODEL.patch_size, image_size=cfg.MODEL.img_size, alpha_t=cfg.TRAIN.alpha_t)
        pred_pixel_values_s, masked_patches_s = learner(inp, masked_indices_s, unmasked_indices_s, num_masked)
        loss2 = F.mse_loss(pred_pixel_values_s, masked_patches_s)

        ######### Cross-Branch Consistency #########
        p1 = learner.encoder.image_patch_size
        p2 = learner.encoder.image_patch_size
        pf = learner.encoder.frame_patch_size
        recons_tokens_l = torch.zeros(cfg.TRAIN.batch_size, patch_num, p1 * p2 * pf, device='cuda:0')
        recons_tokens_l[batch_range, masked_indices_l] = pred_pixel_values_l
        recons_tokens_s = torch.zeros(cfg.TRAIN.batch_size, patch_num, p1 * p2 * pf, device='cuda:0')
        recons_tokens_s[batch_range, masked_indices_s] = pred_pixel_values_s
        active_map = (torch.mean(recons_tokens_l, dim=2) != 0) & (torch.mean(recons_tokens_s, dim=2) != 0)
        l2_c1 = ((recons_tokens_l - recons_tokens_s.detach()) ** 2).mean(dim=2, keepdim=False)
        l2_c1 = l2_c1.mul_(active_map).sum() / (active_map.sum() + 1e-8)
        l2_c2 = ((recons_tokens_s - recons_tokens_l.detach()) ** 2).mean(dim=2, keepdim=False)
        l2_c2 = l2_c2.mul_(active_map).sum() / (active_map.sum() + 1e-8)
        loss3 = 0.5 * l2_c1 + 0.5 * l2_c2

        ######### Total Loss #########
        loss = loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()
        learner_ema.update(learner)

        sum_loss += loss.item()
        sum_time += time.time() - t1

        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info(
                    'step %d, loss = %.6f (wt: *1, et: %.2f sec, rd: %.2f min)'
                    % (iters, sum_loss, sum_time,
                       (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info(
                    'step %d, loss = %.6f (wt: *1, et: %.2f sec, rd: %.2f min)' \
                    % (iters, sum_loss / cfg.TRAIN.display_freq * 1, sum_time, \
                       (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = %d, loss = %.6f' % (iters, sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0

        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            visual_2d(learner, inp, att, cfg.cache_path, iters)

        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                      'model_weights': learner.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'learner.ckpt'))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)

    f_loss_txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='orgmim', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider = load_dataset(cfg)
        init_iters = 0
        loop(cfg, train_provider, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')
