
# coding: utf-8
import argparse
import os
import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from model_gt import DM2FNetv8,Discriminator
from tools.config import TRAIN_ITS_ROOT, TEST_SOTS_ROOT
from datasets import ItsDataset, SotsDataset,OtsDataset
from tools.utils import AvgMeter, check_mkdir

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='1', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='./ckpt1', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='RESIDE_ITS',
        help='experiment name.')
    args = parser.parse_args()

    return args


cfgs = {
    'use_physical': True,
    'iter_num': 80000,
    'train_batch_size': 16,
    'last_iter': 0,
    'lr': 4e-5,#5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    #'snapshot': 'iter_10000_loss_0.01761_lr_0.000386',
    'snapshot':'iter_75000_loss_0.00971_lr_0.000041',
    'val_freq': 5000,
    'crop_size': 256,
    
}


def main():
    net = DM2FNetv8().cuda().train()
    # net = nn.DataParallel(net)
    Dnet=Discriminator().cuda().train()
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] != 'bias' and param.requires_grad],
         'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    ])
    Doptimizer = optim.Adam([
        {'params': [param for name, param in Dnet.named_parameters()
                    if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in Dnet.named_parameters()
                    if name[-4:] != 'bias' and param.requires_grad],
         'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    ])

    if len(cfgs['snapshot']) > 0:
        print('training resumes from \'%s\'' % cfgs['snapshot'])
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                    args.exp_name, cfgs['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                          args.exp_name, cfgs['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * cfgs['lr']
        optimizer.param_groups[1]['lr'] = cfgs['lr']

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(cfgs) + '\n\n')

    train(net, Dnet,optimizer,Doptimizer)


def train(net,dnet,optimizer,doptimizer):
    curr_iter = cfgs['last_iter']
    real_label=torch.ones([cfgs['train_batch_size'],1]).cuda()
    fake_label=torch.zeros([cfgs['train_batch_size'],1]).cuda()
    while curr_iter <= cfgs['iter_num']:
        train_loss_record = AvgMeter()
        loss_x_jf_record, loss_x_j0_record = AvgMeter(), AvgMeter()
        loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
        loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()
        loss_t_record, loss_a_record = AvgMeter(), AvgMeter()
        loss_d_record=AvgMeter()

        for data in train_loader:
            
            optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']
            optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']
            doptimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']
            doptimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']
            haze, gt_trans_map, gt_ato, gt, _ = data
           

            #gnettrain
            batch_size = haze.size(0)

            haze = haze.cuda()
            gt_trans_map = gt_trans_map.cuda()
            gt_ato = gt_ato.cuda()
            gt = gt.cuda()
            
            optimizer.zero_grad()

            x_jf, x_j0, x_j1, x_j2, x_j3, x_j4, t, a = net(haze)

            loss_x_jf = criterion(x_jf, gt)
            loss_x_j0 = criterion(x_j0, gt)
            loss_x_j1 = criterion(x_j1, gt)
            loss_x_j2 = criterion(x_j2, gt)
            loss_x_j3 = criterion(x_j3, gt)
            loss_x_j4 = criterion(x_j4, gt)

            loss_t = criterion(t, gt_trans_map)
            loss_a = criterion(a, gt_ato)
            ##set dnet requires_grad
            net.requires_grad_(False)
            dnet.requires_grad_(True)
            dnet.zero_grad()
            #real image            
            real=torch.max(dnet(gt),dim=1).values
            real_output=real.reshape(-1,1)
            d_real_loss=bce_loss(real_output,real_label)
            d_real_loss.backward()
            #fake image
            fake_image=x_jf.detach()
            fake=torch.max(dnet(fake_image),dim=1).values
            fake_output=fake.reshape(-1,1)
            d_fake_loss=bce_loss(fake_output,fake_label)
            d_fake_loss.backward()    
            doptimizer.step()
            loss_d=d_fake_loss+d_real_loss

            dnet.requires_grad_(False)
            net.requires_grad_(True)
            net.zero_grad()
            ##ganloss
            
            real_o=torch.max(dnet(x_jf),dim=1).values
            real_out_o=real_o.reshape(-1,1)
            gan_loss=bce_loss(real_out_o,real_label)
            #loss compute
            loss = loss_x_jf + loss_x_j0 + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4 \
                   + 10 * loss_t + loss_a+ 0.0001*gan_loss
            # loss=gan_loss
            loss.backward()

            optimizer.step()


            # update recorder
            train_loss_record.update(loss.item(), batch_size)

            loss_x_jf_record.update(loss_x_jf.item(), batch_size)
            loss_x_j0_record.update(loss_x_j0.item(), batch_size)
            loss_x_j1_record.update(loss_x_j1.item(), batch_size)
            loss_x_j2_record.update(loss_x_j2.item(), batch_size)
            loss_x_j3_record.update(loss_x_j3.item(), batch_size)
            loss_x_j4_record.update(loss_x_j4.item(), batch_size)

            loss_t_record.update(loss_t.item(), batch_size)
            loss_a_record.update(loss_a.item(), batch_size)
            loss_d_record.update(loss_d.item(), batch_size)


            curr_iter += 1

            log = '[iter %d], [train loss %.5f],[loss_d %.5f], [loss_x_fusion %.5f], [loss_x_phy %.5f], [loss_x_j1 %.5f], ' \
                  '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f], [loss_t %.5f], [loss_a %.5f], ' \
                  '[lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, loss_d_record.avg,loss_x_jf_record.avg, loss_x_j0_record.avg,
                   loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg,
                   loss_t_record.avg, loss_a_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % cfgs['val_freq'] == 0:
                validate(net, curr_iter, optimizer)

            if curr_iter > cfgs['iter_num']:
                break


def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()

    with torch.no_grad():
        for data in tqdm(val_loader):
            haze, gt, _ = data

            haze = haze.cuda()
            gt = gt.cuda()

            dehaze = net(haze)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))

    snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[1]['lr'])
    print('[validate]: [iter %d], [loss %.5f]' % (curr_iter + 1, loss_record.avg))
    torch.save(net.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True
    torch.cuda.set_device(0)

    train_dataset = ItsDataset(TRAIN_ITS_ROOT, True, cfgs['crop_size'])
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                              shuffle=True, drop_last=True)

    val_dataset = SotsDataset(TEST_SOTS_ROOT)
   # val_dataset=OtsDataset(TEST_SOTS_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=cfgs['train_batch_size'])

    criterion = nn.L1Loss().cuda()
    bce_loss=nn.BCELoss().cuda()
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')

    main()