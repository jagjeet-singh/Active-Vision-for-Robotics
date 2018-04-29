import datetime
import os
import sys
import pdb
import argparse
sys.path.append(os.path.join(os.getcwd(),os.pardir, os.pardir))
import random
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from models import *
from datasets import active_vision_dataloader
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from logger import Logger

use_cuda = torch.cuda.is_available()
print('#### Using cuda ####') if use_cuda else print('#### Not using cuda ####')
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

train_args = {}
cudnn.benchmark = True


# hyperparams = {
#     'epoch_num': 300,
#     'lr': 1e-10,
#     'weight_decay': 1e-4,
#     'momentum': 0.95,
#     'lr_patience': 100,  # large patience denotes fixed lr
#     'snapshot': '',  # empty string denotes learning from scratch
#     'print_freq': 20,
#     'val_save_to_img_file': False,
#     'val_img_sample_rate': 0.1  # randomly sample some validation results to display
# }

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',type=float, default=1e-10,help="Learning rate")
    parser.add_argument('--epoch-num',type=int, default=300,help="Number of epochs")
    parser.add_argument('--weight-decay',type=float, default=1e-4,help="Weight decay")
    parser.add_argument('--momentum',type=float, default=0.95,help="Momentum")
    parser.add_argument('--lr-patience',type=float, default=100,help="large patience denotes fixed lr")
    parser.add_argument('--snapshot', default='', type=str, help='path to latest checkpoint (empty string denotes learning from scratch)')
    parser.add_argument('--print-freq',type=int, default=20,help="Print frequency")
    parser.add_argument('--val-img-sample-rate',type=float, default=0.1,help="randomly sample some validation results to display")
    parser.add_argument('--val-freq',type=int, default=5,help="Frequency of plotting on Tensorboard")
    parser.add_argument('--plot-freq',type=int, default=100,help="Frequency of plotting on Tensorboard")
    parser.add_argument('--hist-freq',type=int, default=500,help="Frequency of plotting histograms on Tensorboard")
    parser.add_argument('--batch-size',type=int, default=1,help="Training batch size")
    parser.add_argument('--without-target',action='store_true',help="Normal segmentation or one-shot with target?")
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--test', action='store_true', help='Test the trained model')
    parser.add_argument('--trainval-split', default=0.8, type=float)
    parser.add_argument('--tb-logdir', default='active-vision', type=str)
    parser.add_argument('--psc_groupid', default='ir5fp2p', type=str)
    parser.add_argument('--psc_userid', default='jagjeets', type=str)
    return parser.parse_args()

def make_dataset(mode, image_folder, mask_folder, target_folder):
    args = parse_arguments()
    if mode == 'train' or mode == 'val':
        image_names = sorted(os.listdir(image_folder))
        mask_names = sorted(os.listdir(mask_folder))
        target_names = sorted(os.listdir(target_folder))
        trainval_size = len(image_names)
        combined = list(zip(image_names, mask_names, target_names))  
        random.shuffle(combined)
        image_names, mask_names, target_names = zip(*combined)
        trainval_idx = int(args.trainval_split*trainval_size)
        train_image_names = image_names[:trainval_idx]
        train_mask_names = mask_names[:trainval_idx]
        train_target_names = target_names[:trainval_idx]
        val_image_names = image_names[trainval_idx:]
        val_mask_names = mask_names[trainval_idx:]
        val_target_names = target_names[trainval_idx:]
        return train_image_names, train_mask_names, train_target_names, val_image_names, val_mask_names, val_target_names
    else:
        image_names = sorted(os.listdir(image_folder))
        target_names = sorted(os.listdir(target_folder))
        return image_names, target_names

def main():
    
    args = parse_arguments()
    # Setting paths as per psc
    storage_path = 'pylon5/'+args.psc_groupid+'/'+args.psc_userid+'/active-vision-storage'
    logger = Logger(storage_path+'av_tblogs', name=args.tb_logdir)
    ckpt_path = storage_path+'ckpt'
    exp_name = 'av_fcn8s'
    image_folder = storage_path+'av_data/images'
    mask_folder = storage_path+'av_data/masks'
    target_folder = storage_path+'av_data/targets'

    if args.without_target:
        net = FCN8s_wot(num_classes=2)
    else:
        net = FCN8s(num_classes=2)
    if use_cuda:
        net.cuda()

    if len(args.snapshot) == 0:
        curr_epoch = 1
        train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print('training resumes from ' + args.snapshot)
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args.snapshot)))
        split_snapshot = args.snapshot.split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                     'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                     'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}

    transform = transforms.Compose([
        transforms.ToTensor()])
    train_image_names, train_mask_names, train_target_names, val_image_names, val_mask_names, val_target_names  = make_dataset('train', image_folder, mask_folder, target_folder)
    train_dataset = active_vision_dataloader.AV(train_image_names, train_mask_names, train_target_names, image_folder, mask_folder, target_folder, mode='train', transform=transform)
    val_dataset = active_vision_dataloader.AV(val_image_names, val_mask_names, val_target_names, image_folder, mask_folder, target_folder, mode='val', transform=transform)
    
    # Setting Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4,drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,num_workers=4, batch_size=args.batch_size,
        drop_last=True, shuffle=True)

    criterion = CrossEntropyLoss2d(size_average=True)
    if use_cuda:
        criterion.cuda()

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args.lr, 'weight_decay': args.weight_decay}
    ], betas=(args.momentum, 0.999))

    if len(args.snapshot) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + args.snapshot)))
        optimizer.param_groups[0]['lr'] = 2 * args.lr
        optimizer.param_groups[1]['lr'] = args.lr

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    # open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(train_args) + '\n\n')

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, min_lr=1e-10, verbose=True)
    for epoch in range(curr_epoch, args.epoch_num + 1):
        val_loss = validate(val_loader, net, criterion, optimizer, epoch, logger)
        train(train_loader, net, criterion, optimizer, epoch, logger)
        if epoch % args.valid_freq == 0:
            val_loss = validate(val_loader, net, criterion, optimizer, epoch, logger, ckpt_path, exp_name)
        scheduler.step(val_loss)

def train(train_loader, net, criterion, optimizer, epoch, logger):
    args = parse_arguments()
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, targets, labels = data
        # Verify this part
        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)
        inputs = Variable(inputs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        targets = Variable(targets.type(FloatTensor))
        # if use_cuda:
        #     inputs = inputs.cuda()
        #     labels = labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs, targets)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == 2
        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data[0], N)

        curr_iter += 1
        logger.scalar_summary(tag='Train/Loss', value=train_loss.avg, step=curr_iter)
        if i%args.hist_freq == 0:
            logger.model_param_histo_summary(model=net, step=curr_iter)
        if (i + 1) % args.print_freq == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg
            ))

def validate(val_loader, net, criterion, optimizer, epoch, logger, ckpt_path, exp_name):
    args = parse_arguments()
    net.eval()
    val_loss = AverageMeter()
    inputs_all, targets_all, gts_all, predictions_all = [], [], [], []
    val_size = len(val_loader)
    for vi, data in enumerate(val_loader):
        inputs, targets, gts = data
        N = inputs.size(0)
        inputs = Variable(inputs.type(FloatTensor), volatile=True)
        targets = Variable(targets.type(FloatTensor), volatile=True)
        gts = Variable(gts.type(LongTensor), volatile=True)

        outputs = net(inputs, targets)
        if use_cuda:
            predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy() 
        else:
            predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).numpy()

        val_loss.update(criterion(outputs, gts).data[0] / N, N)

        if random.random() > args.val_img_sample_rate:
            inputs_all.append(inputs.data.squeeze_(0).cpu())
            targets_all.append(targets.data.squeeze_(0).cpu())
            gts_all.append(gts.data.squeeze_(0).cpu().numpy())
            predictions_all.append(predictions)

        if vi%args.plot_freq == 0:
            logger.image_summary(tag='epoch'+str(epoch)+'/iter'+str(vi)+'/image', images=[inputs.data.squeeze_(0)], step=epoch*val_size+vi)
            logger.image_summary(tag='epoch'+str(epoch)+'/iter'+str(vi)+'/gt_mask', images=[gts.data.squeeze_(0)], step=epoch*val_size+vi)
            logger.image_summary(tag='epoch'+str(epoch)+'/iter'+str(vi)+'/pred_mask', images=[predictions], step=epoch*val_size+vi)

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, 2)
    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr']
        )
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

    print('--------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))

    print('--------------------------------------------------------------------')

    logger.scalar_summary(tag='Val/loss', value=val_loss.avg, step=epoch)
    logger.scalar_summary(tag='Val/acc', value=acc, step=epoch)
    logger.scalar_summary(tag='Val/acc_cls', value=acc_cls, step=epoch)
    logger.scalar_summary(tag='Val/mean_iu', value=mean_iu, step=epoch)
    logger.scalar_summary(tag='Val/fwavacc', value=fwavacc, step=epoch)
    logger.scalar_summary(tag='lr', value=optimizer.param_groups[1]['lr'], step=epoch)

 
    net.train()
    return val_loss.avg


if __name__ == '__main__':
    main()