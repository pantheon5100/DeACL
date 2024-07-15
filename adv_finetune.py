from __future__ import print_function

import argparse
import math
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from autoattack import AutoAttack
from trades.trades import trades_loss, _pgd_whitebox
from torch import nn, optim
from torchvision import datasets, transforms

# from models.resnet_cifar import ResNet18
# from solo.models.resnet_cifar import ResNet18, ResNet50
from solo.models.wide_resnet import wide_resnet28w10
from solo.models.model_with_linear import ModelwithLinear, LinearClassifier
from solo.models.resnet_add_normalize import resnet18_NormalizeInput

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def set_loader(opt):
    # construct data loader
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                     transform=train_transform,
                                     download=True)
    val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                   train=False,
                                   transform=val_transform)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(
            train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--mode', type=str, default='aff', choices=['slf', 'aff', 'alf'],
                        help='mode of fine-tuning')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=2e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10'], help='dataset')
    # other setting
    parser.add_argument('--ckpt', type=str, default='trained_models/res18_simclr-cifar10-offline-x4h7cp45-ep=99.ckpt',
                        help='path to pre-trained model')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='print frequency')
    
    # adversarial setting
    parser.add_argument('--epsilon', type=float, default=8., help='epsilon')
    parser.add_argument('--num_steps_train', type=int, default=10, help='num_steps')
    parser.add_argument('--num_steps_test', type=int, default=20, help='num_steps')
    parser.add_argument('--step_size', type=float, default=2., help='step_size')
    parser.add_argument('--random_start', type=bool, default=True, help='random_start')
    
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './data/cifar10'
    opt.n_cls = 10
    
    # convert to float
    opt.epsilon = opt.epsilon / 255.
    opt.step_size = opt.step_size / 255.

    return opt


# # PGD attack model
# class AttackPGD(nn.Module):
#     def __init__(self, model, classifier, config):
#         super(AttackPGD, self).__init__()
#         self.model = model
#         self.classifier = classifier
#         self.rand = config['random_start']
#         self.step_size = config['step_size']
#         self.epsilon = config['epsilon']
#         assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'

#     def forward(self, inputs, targets, train=True):
#         x = inputs.detach()
#         if self.rand:
#             x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
#         if train:
#             num_step = 10
#         else:
#             num_step = 20
#         for i in range(num_step):
#             x.requires_grad_()
#             with torch.enable_grad():
#                 features = self.model(x)
#                 logits = self.classifier(features)
#                 loss = F.cross_entropy(logits, targets, size_average=False)
#             grad = torch.autograd.grad(loss, [x])[0]
#             x = x.detach() + self.step_size * torch.sign(grad.detach())
#             x = torch.min(torch.max(x, inputs - self.epsilon),
#                           inputs + self.epsilon)
#             x = torch.clamp(x, 0, 1)
#         features = self.model(x)
#         return self.classifier(features), x


def set_model(opt):
    if "res50" in opt.ckpt:
        model = ResNet50()
        classifier = LinearClassifier(feat_dim=2048, num_classes=opt.n_cls)
    elif "wideres28_10" in opt.ckpt:
        model = wide_resnet28w10()
        classifier = LinearClassifier(feat_dim=model.inplanes, num_classes=opt.n_cls)
    else:
        model = resnet18_NormalizeInput()
        model.fc = nn.Identity()
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=2, bias=False
        )
        model.maxpool = nn.Identity()
        classifier = LinearClassifier(feat_dim=512, num_classes=opt.n_cls)

    print('loading from {}'.format(opt.ckpt))
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    state_dict_load = {}
    for k,v in state_dict.items():
        if k.startswith('backbone.'):
            state_dict_load[k.replace('backbone.', '')] = v.clone()
    model.load_state_dict(state_dict_load, strict=True)

    model = model.cuda()
    classifier = classifier.cuda()
    
    def model_forward(x):
        x = model(x)
        x = classifier(x)
        return x

    # cudnn.benchmark = True
    

    # build loss function definition
    if opt.mode == 'slf':
        criterion = torch.nn.CrossEntropyLoss()
        def loss_function(x, y):
            with torch.no_grad():
                features = model(x)
            logits = classifier(features)
            loss = criterion(logits, y)
            return loss, logits
        
        # build optimizer
        params = list(classifier.parameters())
        # set the model to be fixed
        for param in model.parameters():
            param.requires_grad = False
        optimizer = optim.SGD(params,
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)

    elif opt.mode == 'aff':
        def loss_function(x, y):
            loss, logits = trades_loss(
                model_forward, model, classifier, x, y, optimizer, 
                step_size=opt.step_size, epsilon=opt.epsilon, perturb_steps=opt.num_steps_train, 
                beta=6.0, distance='l_inf'
                )
            return loss, logits
        
        # build optimizer
        params = list(classifier.parameters()) + list(model.parameters())
        optimizer = optim.SGD(params,
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    
    elif opt.mode == 'alf':
        def loss_function(x, y):
            loss, logits = trades_loss(
                model_forward, model, classifier, x, y, optimizer, 
                step_size=opt.step_size, epsilon=opt.epsilon, perturb_steps=opt.num_steps_train, 
                beta=6.0, distance='l_inf'
                )
            return loss, logits
        
        # build optimizer
        params = list(classifier.parameters())
        # set the model to be fixed
        for param in model.parameters():
            param.requires_grad = False
        optimizer = optim.SGD(params,
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    
    else:
        raise ValueError(f"Unknown mode: {opt.mode}")
        
    return model, classifier, model_forward, loss_function, optimizer


def train(train_loader, loss_function, optimizer, epoch, opt):
    """one epoch training"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        loss, output = loss_function(images, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train Step: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, idx + 1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg

@torch.no_grad()
def validate(val_loader, model_forward, opt):
    """validation"""
    criterion = torch.nn.CrossEntropyLoss()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    top1_clean = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(val_loader):
        images = images.float().cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]

        # forward
        out_clean, out_pgd = _pgd_whitebox(model_forward, images, labels, opt.epsilon, opt.num_steps_test, opt.step_size, opt.random_start, 'cuda')

        # update metric
        loss = criterion(out_pgd, labels)
        losses.update(loss.item(), bsz)
        acc1_clean, acc5_clean = accuracy(out_clean, labels, topk=(1, 5))
        top1_clean.update(acc1_clean[0], bsz)
        acc1, acc5 = accuracy(out_pgd, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Test Step: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 Clean {top1_clean.val:.4f} ({top1_clean.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        idx, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1, top1_clean=top1_clean))

    return losses.avg, top1.avg, top1_clean.avg


def adjust_lr(lr, optimizer, epoch):
    if epoch >= 15:
        lr /= 10
    if epoch >= 20:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model, loss function, and optimizer
    model, classifier, model_forward, loss_function, optimizer = set_model(opt)


    # -----------------------------------------------
    # Fine-tuning 
    # -----------------------------------------------
    for epoch in range(1, opt.epochs + 1):
        adjust_lr(opt.learning_rate, optimizer, epoch-1)
        # train for one epoch
        time1 = time.time()
        model.train()
        classifier.train()
        loss, acc = train(train_loader, loss_function,
                          optimizer, epoch, opt)
        time2 = time.time()
        print(f'Train Epoch: [{epoch}/{opt.epochs}], epoch time: {time2 - time1:.2f}, loss: {loss:.2f}, acc: {acc:.2f}\n')
        
        if epoch % opt.eval_freq == 0:
            # eval for one epoch
            model.eval()
            classifier.eval()
            loss, val_acc, val_acc_clean = validate(
                val_loader, model_forward, opt)
            print(f'Validate [{epoch}/{opt.epochs}] \n*Loss: {loss:.2f} \n*Val acc: {val_acc:.2f} \n*Val acc clean: {val_acc_clean:.2f}\n')
    # -----------------------------------------------

    
    # -----------------------------------------------
    # Robustness and clean sample evaluation
    # -----------------------------------------------
    print('\nEvaluating robustness and clean sample accuracy...')
    model.eval()
    classifier.eval()
    loss, val_acc, val_acc_clean = validate(val_loader, model_forward, opt)
    print(f'*Loss: {loss:.2f} \n*Robust acc: {val_acc:.2f} \n*Clean accuracy: {val_acc_clean:.2f}\n')
    
    ckpt_name = os.path.basename(opt.ckpt).split('.')[0]
    with open(os.path.join(os.path.dirname(opt.ckpt), f"{opt.mode}_Robust_Test-{ckpt_name}.log"), "a") as f:
        f.write(f"Final robust acc: {val_acc}.\n")
        f.write(f"Final clean acc: {val_acc_clean}.\n")
    # -----------------------------------------------
    
    
    # -----------------------------------------------
    # AutoAttack evaluation
    # -----------------------------------------------
    print('\nEvaluating AutoAttack...')
    log_path = os.path.join(os.path.dirname(opt.ckpt), f"{opt.mode}_AutoAttack-{ckpt_name}.log")
    adversary = AutoAttack(model_forward, 
                           norm="Linf", eps=opt.epsilon, 
                           log_path=log_path, version="standard", seed=0)
    l = [x for (x, y) in val_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in val_loader]
    y_test = torch.cat(l, 0)
    with torch.no_grad():
        adversary.run_standard_evaluation(x_test, y_test, bs=256)
    # -----------------------------------------------


if __name__ == '__main__':
    main()
