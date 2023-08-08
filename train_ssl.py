import argparse
import timm.utils
# transformation
from timm.data.readers.class_map import load_class_map
from PIL import Image
from torchvision import transforms
from randaugment import RandAugmentMC
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from pathlib import Path
# model
import torch
from timm import create_model
# optimizer
from torch import optim
from torch.nn import functional as F
# utils
from pprint import pprint
from utils import AverageMeter, accuracy
from tqdm import tqdm
import time
import logging
import shutil
import os
import numpy as np
from timm.utils import CheckpointSaver
from pprint import pprint
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
# custom module
from custom_module import CosineAnnealingWarmUpRestarts

logger = logging.getLogger(__name__)
best_acc = 0

parser = argparse.ArgumentParser(description='SSL - FixMatch')

## MODEL SETTINGS
group = parser.add_argument_group('Model setting')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet50")')
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Initialize model from this checkpoint (default: none)')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
group.add_argument('--class-map', type=str, metavar='MAP',
                   help='class map file')
group.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                   help='Input batch size for training (default: 64)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override (default: None)')
group.add_argument('--epochs', type=int, default=100, metavar='EPOCH',
                   help='epoch')
## DATASET SETTINGS
group = parser.add_argument_group('Dataset setting')
group.add_argument('--labeled-img-dir', type=str, metavar='DIR',
                   help='labeled image directory')
group.add_argument('--unlabeled-img-dir', type=str, metavar='DIR',
                   help='unlabeled image directory')
group.add_argument('--test-img-dir', type=str, metavar='DIR',
                   help='test image directory')
group.add_argument('--mean', type=float, nargs=3, default=[0.4594, 0.5552, 0.3630], metavar='MEAN',
                   help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs=3, default=[0.1445, 0.1309, 0.1556], metavar='STD',
                   help='Override std deviation of dataset')
group.add_argument('--img-size', type=int, metavar='SIZE',
                   help='img size')
group.add_argument('--eval-step', type=int, metavar='STEP', default=50,
                   help='evaluation step')
# OPTIMIZER SETTINGS
group = parser.add_argument_group('Optimizer setting')
group.add_argument('--opt', type=str, metavar='OPT',
                   help='set optimizer')
group.add_argument('--lr', type=float, metavar='LR', default=0.03,
                   help='learning rate, default=0.03')
group.add_argument('--momentum', type=float, metavar='MNT', default=0.9,
                   help='momentum, default=0.9')
group.add_argument('--wdecay', type=float, metavar='WD', default=5e-4,
                   help='weight decay factor, default=5e-4')
group.add_argument('--nesterov', action='store_true',
                   help='use nesterov')
group.add_argument('--lambda_u', type=float, metavar='λ', default=1,
                   help='unlabeled loss weight for FixMatch, default=1')
group.add_argument('--T', default=1, type=float,
                    help='pseudo label temperature, default=1')
group.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold, default=0.95')
group.add_argument('--mu', default=7, type=int,
                    help='coefficient of unlabeled batch size, default=7')
# MISCELLANEOUS SETTINGS
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging')
group.add_argument('--checkpoint-hist', type=int, default=1, metavar='N',
                   help='number of checkpoints to keep (default: 1)')
group.add_argument("--rank", default=0, type=int)
group.add_argument("--save-path", type=str, default='output',
                   help='path to save')
group.add_argument("--save-best", type=str, default='model_best.pth.tar',
                   help='best model name')
group.add_argument("--save-last", type=str, default='last.pth.tar',
                   help='last model name')
group.add_argument("--num-workers", type=int, default=4,
                   help='the number of worker, default=4')
group.add_argument("--device", type=str, default='cuda',
                   help='device')
group.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
group.add_argument('--no-progress', action='store_true',
                    help="don't use progress bar")

class ImageFolder(Dataset):
    def __init__(self, root, transform=None, class_map=None) -> None:
        super().__init__()
        self.root = root
        self.tranform=transform
        self.class_map = class_map
        if self.class_map:
            self.class2idx = load_class_map(self.class_map)
        else:
            # self.class2idx = {class_name.name:idx for idx, class_name in enumerate(Path(self.root).iterdir())}
            self.class2idx = None

    def __len__(self):
        img_files = [
        *list(Path(self.root).rglob('*.jpg')),
        *list(Path(self.root).rglob('*.jpeg')),
        *list(Path(self.root).rglob('*.JPG')),
        *list(Path(self.root).rglob('*.jpeg'))
        ]
        return len(img_files)
    
    def __getitem__(self, index):
        img_files = [
        *list(Path(self.root).rglob('*.jpg')),
        *list(Path(self.root).rglob('*.jpeg')),
        *list(Path(self.root).rglob('*.JPG')),
        *list(Path(self.root).rglob('*.jpeg'))
        ]
        img = Image.open(img_files[index])
        if self.class2idx:
            label = self.class2idx[img_files[index].parent.name]
        else:
            label=0
        if self.tranform:
            img = self.tranform(img)
        return img, label
    
class TransformFixMatch():
    def __init__(self, size, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def interleave(x, size):
    s = list(x.shape)
    print(s)
    print(x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:]).shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def main():
    args=parser.parse_args()

    if args.local_rank == -1:
        device = torch.device('cuda', 0)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    # Set seed
    timm.utils.random_seed(args.seed, args.rank)
    # define weak, strong tranformation
    weak_tf = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.img_size,
                                padding=int(args.img_size*0.125),
                                padding_mode='reflect'),
        transforms.Normalize(mean=args.mean, std=args.std)
    ])
    strong_tf = transforms.Compose([
            transforms.Resize((args.img_size,args.img_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.img_size,
                                  padding=int(args.img_size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.Normalize(mean=args.mean, std=args.std)
    ])
    test_tf = transforms.Compose([
            transforms.Resize((args.img_size,args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
        ])
    # Load labeled dataset, unlabeled dataset using waek, strong transformation
    labeled_train_dataset = ImageFolder(
        root=args.labeled_img_dir,
        transform=weak_tf,
        class_map=args.class_map
        )
    unlabeled_train_dataset = ImageFolder(
        root=args.unlabeled_img_dir,
        transform=TransformFixMatch(size=args.img_size, mean=args.mean, std=args.std),
        )
    test_dataset = ImageFolder(
        root=args.test_img_dir,
        transform=test_tf,
        class_map=args.class_map
        )
    
    # Get dataloader
    train_sampler = RandomSampler
    test_sampler = SequentialSampler
    labeled_train_dataloader=DataLoader(labeled_train_dataset, batch_size=args.batch_size, sampler=train_sampler(labeled_train_dataset), num_workers=args.num_workers)
    unlabeled_train_dataloader=DataLoader(unlabeled_train_dataset, batch_size=args.batch_size*args.mu, sampler=train_sampler(unlabeled_train_dataset), num_workers=args.num_workers)
    test_dataloader=DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler(test_dataset), num_workers=args.num_workers)

    # Load model
    model = create_model(args.model, checkpoint_path=args.initial_checkpoint, num_classes=args.num_classes, pretrained=args.pretrained)
    
    # if args.local_rank == 0:
    #     torch.distributed.barrier()
    model.to(args.device)
    
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any( \
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any( \
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Set optimizer
    #  optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay, nesterov=args.nesterov)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=500)

    #### use custom scheduler
    optimizer = optim.AdamW(grouped_parameters, lr=0, weight_decay=args.wdecay)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, T_0=500, T_mult=1, eta_max=args.lr, T_up=5, gamma=0.5)

    model.zero_grad()
    # Set saver
    saver = CheckpointSaver(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=args.save_path,
        max_history=args.checkpoint_hist
    )

    train(args, labeled_train_dataloader, unlabeled_train_dataloader, test_dataloader,
          model, optimizer, scheduler, saver=saver)

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, scheduler, saver):
    print('evaluate initial model')
    pre_loss, pre_acc = test(args, test_loader, model)

    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                # inputs_x, targets_x = labeled_iter.next()
                # error occurs ↓
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                # inputs_x, targets_x = labeled_iter.next()
                # error occurs ↓
                inputs_x, targets_x = next(labeled_iter)

            try:
                # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                # error occurs ↓
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                # error occurs ↓
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            if inputs_x.shape[0]+inputs_u_w.shape[0]+inputs_u_s.shape[0] % 2*args.mu+1 == 0:
                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            else:
                inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            if inputs_x.shape[0]+inputs_u_w.shape[0]+inputs_u_s.shape[0] % 2*args.mu+1 == 0:
                logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            ### DEBUG
            #  print('=============')
            #  denorm_tf = transforms.Compose([
                #  transforms.Normalize(mean=[0, 0, 0], std=[1/x for x in args.std]),
                #  transforms.Normalize(mean=[-x for x in args.mean], std=[1, 1, 1]),
                #  ])
            #  print(pseudo_label)
            #  for target, prob in zip(targets_u.detach().tolist(), max_probs.detach().tolist()):
                #  print(f'{target} / {prob:.4f}')
            #  plt.imshow(make_grid(denorm_tf(inputs.detach().cpu()), normaliz=True).permute(1,2,0))
            #  plt.show()
            #########

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + args.lambda_u * Lu

            # if args.amp:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        # if args.use_ema:
        #     test_model = ema_model.ema
        # else:
        test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            # if args.use_ema:
            #     ema_to_save = ema_model.ema.module if hasattr(
            #         ema_model.ema, "module") else ema_model.ema
            if saver and args.save_path:
                acc = 100.*test_acc
                if acc > best_acc:
                    print('Saving to best model')
                    if not os.path.isdir(args.save_path):
                        os.mkdir(args.save_path)
                    saver.save_checkpoint(epoch, metric=acc)
                if epoch==args.epochs-1:
                    print('Saving to last model')
                    if not os.path.isdir(args.save_path):
                        os.mkdir(args.save_path)
                    saver.save_checkpoint(epoch, metric=acc)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    #  forDebug = [] # DEBUG
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            #  forDebug.append(prec1.item()) # DEBUG
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    #  pprint(forDebug) # DEBUG
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg

if __name__=='__main__':
    main()
