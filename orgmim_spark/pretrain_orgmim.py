from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import logging
import argparse
import numpy as np
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict
import torch
import torch.nn as nn
from pretrain_provider import Provider
from utils.seed import setup_seed
import torch.nn.functional as F
from utils.visual2d import visual_2d
from timm.utils import ModelEma
import cv2

def tuple_to_list(obj):
    if isinstance(obj, tuple):
        return [tuple_to_list(x) for x in obj]
    elif isinstance(obj, list):
        return [tuple_to_list(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: tuple_to_list(v) for k, v in obj.items()}
    else:
        return obj

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M',
            filename=path,
            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
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

    from mim.encoder3D import SparseEncoder
    from mim.stunet_head import STUNet
    from mim.spark_orgmim import SparK
    from mim.decoder3D import LightDecoder

    # Initialize the model components
    head = STUNet(1, 1, depth=tuple_to_list(cfg.MODEL.depth), dims=tuple_to_list(cfg.MODEL.dims),
                  pool_op_kernel_sizes=tuple_to_list(cfg.MODEL.pool_op_kernel_sizes), conv_kernel_sizes=tuple_to_list(cfg.MODEL.conv_kernel_sizes),
                  enable_deep_supervision=True)
    input_size = (cfg.MODEL.img_size, cfg.MODEL.img_size, cfg.MODEL.img_size)

 
    enc = SparseEncoder(head, input_size=input_size, sbn=False)
    dec = LightDecoder(enc.downsample_ratio, sbn=False, width=512, out_channel=1)

    learner = SparK(
        sparse_encoder=enc, dense_decoder=dec, mask_ratio=cfg.TRAIN.mask_ratio,
        densify_norm='in'
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

    learner = learner.cuda()
    learner_ema = ModelEma(learner, decay=0.999, device='cuda:0', resume='')
    learner_ema.decay = 0.999

    optimizer = torch.optim.AdamW(learner.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999))


    while iters <= cfg.TRAIN.total_iters:
        learner.train()
        iters += 1
        t1 = time.time()
        inp, att = train_provider.next()
        inp, att = inp.cuda(), att.cuda()
        optimizer.zero_grad()
        ######### Loss Branch #########
        recon_loss_list = []
        non_active_list = []
        for _ in range(cfg.TRAIN.loss_forward):
            mask_n = learner.mask(cfg.TRAIN.batch_size, 'cuda:0')
            with torch.no_grad():
                inp1, rec1 = learner_ema.ema(inp, active_b1ff=mask_n, vis=False)
                l2_loss = ((rec1 - inp1) ** 2).mean(dim=2, keepdim=False)
                non_active_n = mask_n.logical_not().int().view(mask_n.shape[0], -1)
                recon_loss_n = l2_loss * non_active_n
                recon_loss_list.append(recon_loss_n)
                non_active_list.append(non_active_n)

        average_result = torch.zeros_like(recon_loss_list[0])
        count_non_zero = torch.zeros_like(recon_loss_list[0])
        for matrix in recon_loss_list:
            non_zero_indices = matrix != 0
            average_result[non_zero_indices] += matrix[non_zero_indices]
            count_non_zero[non_zero_indices] += 1
        average_result[count_non_zero > 0] /= count_non_zero[count_non_zero > 0]
        non_active = torch.max(torch.stack(non_active_list), dim=0).values
        recon_loss = average_result * non_active
        mask2 = learner_ema.ema.generate_mask_alm(recon_loss, step=iters, total_step=cfg.TRAIN.total_iters, alpha_t=cfg.TRAIN.alpha_t)
        mask2 = mask2.to('cuda:0', non_blocking=True)
        non_active2 = mask2.logical_not().int().view(mask2.shape[0], -1)
        inpp2, recc2 = learner(inp, active_b1ff=mask2, vis=False)
        loss1, _ = learner.forward_loss(inpp2, recc2, mask2)
        ######### Semantic Branch #########
        mask3 = learner.generate_mask_mam(att, step=iters, total_step=cfg.TRAIN.total_iters, alpha_t=cfg.TRAIN.alpha_t)
        inpp3, recc3 = learner(inp, active_b1ff=mask3, vis=False)
        non_active3 = mask3.logical_not().int().view(mask3.shape[0], -1)
        loss2, _ = learner.forward_loss(inpp3, recc3, mask3)
        ######### Cross-Branch Consistency #########
        l2_c1 = ((recc2 - recc3.detach()) ** 2).mean(dim=2, keepdim=False)
        l2_c1 = l2_c1.mul_(non_active2 * non_active3).sum() / ((non_active2 * non_active3).sum() + 1e-8) 
        l2_c2 = ((recc3 - recc2.detach()) ** 2).mean(dim=2, keepdim=False)
        l2_c2 = l2_c2.mul_(non_active2 * non_active3).sum() / ((non_active2 * non_active3).sum() + 1e-8) 
        loss3 = 0.5 * l2_c1 + 0.5 * l2_c2
        ######### Loss Cat. #########
        loss = loss1 + loss2 + loss3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 12).item()
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
            f_loss_txt.write('step = %d, loss = %.6f'% (iters, sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            visual_2d(learner, inp, cfg.cache_path, iters)
        # save
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


    
    