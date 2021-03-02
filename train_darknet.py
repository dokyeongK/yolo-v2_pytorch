import argparse
import visdom
from dataset.voc_dataset import VOC_Dataset
from dataset.coco_dataset import COCO_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from loss import Yolo_Loss
import os
from torch.optim.lr_scheduler import StepLR
import time
from model import darknet19
from model.yolo_darknet import YOLOV2
from utils import *

def train(epoch, vis, train_loader, model, criterion, optimizer, scheduler, save_path, save_file_name):

    print('Training of epoch [{}]'.format(epoch))
    tic = time.time()
    model.train()

    for idx, datas in enumerate(train_loader):

        images = datas[0]
        boxes = datas[1]
        labels = datas[2]

        images = images.cuda() # B 3 416 416
        boxes = [b.cuda() for b in boxes] # images boxes info len : batch
        labels = [l.cuda() for l in labels] # images class info len : batch

        preds = model(images) # B 125 13 13
        preds = preds.permute(0, 2, 3, 1) # B 13 13 125

        loss, losses = criterion(preds, boxes, labels) # boxes, labels : gt

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        toc = time.time() - tic

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # for each steps
        if idx % 100 == 0:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'xy_Loss: {xy:.4f}\t'
                  'wh_Loss: {wh:.4f}\t'
                  'conf_Loss: {conf:.4f}\t'
                  'no_conf_Loss: {noconf:.4f}\t'
                  'class_Loss: {cls:.4f}\t'
                  'Learning rate: {lr:.7f} s \t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader),
                          xy=losses[0].item(),
                          wh=losses[1].item(),
                          conf=losses[2].item(),
                          noconf=losses[3].item(),
                          cls=losses[4].item(),
                          lr=lr,
                          time=toc))

            if vis is not None:
                vis.line(X=torch.ones((1, 6)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss, losses[0], losses[1], losses[2], losses[3], losses[4]]).unsqueeze(
                             0).cpu(),
                         win='train_loss',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Total Loss', 'xy_loss', 'wh_loss', 'conf_loss', 'no_conf_loss',
                                           'cls_loss']))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if scheduler is not None:
        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}
    else:
        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, os.path.join(save_path, save_file_name) + '.{}.pth.tar'.format(epoch))

def main():
    # ================================
    # Setting Argumentation
    # ================================

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--backbone_pretrain', type=bool, default=True)
    parser.add_argument('--model_pretrain', type=bool, default=False)

    parser.add_argument('--save_file_name', type=str, default='yolo_v2_training')
    parser.add_argument('--conf_thres', type=float, default=0.01)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--use_visdom', type=bool, default=True)

    parser.add_argument('--dataset_type', type=str, default='voc', help='which dataset you want to use VOC or COCO')

    opts = parser.parse_args()
    print(opts)

    print("Use Visdom : ", opts.use_visdom)

    if opts.use_visdom :
        vis = visdom.Visdom()

    # ================================
    # Data Loader (root type)
    # [VOC]
    #  Train : /VOCDevkit/TRAIN/VOC2007/JPEGImages & /VOCDevkit/TRAIN/VOC2007/Annotations
    #  Test  : /VOCDevkit/TEST/VOC2007/JPEGImages & /VOCDevkit/TEST/VOC2007/Annotations

    # [COCO]
    #  Train : /COCO/images/train2017 & /COCO/annotations/instances_train2017.json
    #  Test  : /COCO/images/val2017 & /COCO/annotations/instances_val2017.json
    # ================================

    data_root = '/mnt/mydisk/hdisk/dataset/VOCdevkit'

    if opts.dataset_type == 'voc':
        train_set = VOC_Dataset(root=data_root, mode='TRAIN')
        test_set = VOC_Dataset(root=data_root, mode='TEST')

    elif opts.dataset_type == 'coco':
        train_set = COCO_Dataset(root=data_root, mode='TRAIN')
        test_set = COCO_Dataset(root=data_root, mode='TEST')

    train_loader = DataLoader(dataset=train_set,
                              batch_size=opts.batch_size,
                              collate_fn=train_set.collate_fn,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=opts.num_workers)

    darknet = darknet19.DarkNet19(num_classes=opts.num_classes)
    model = YOLOV2(darknet=darknet).cuda()

    if opts.backbone_pretrain :
        backbone_weights_file_root = 'yolo-voc.weights'
        print(" Darknet19 (used pre-train weights) : ", backbone_weights_file_root)
        weights_loader = WeightLoader()
        weights_loader.load(darknet, backbone_weights_file_root, backbone=True)

    if opts.model_pretrain:
        # load-weights (include backbone) / .weights file
        model_weights_file_root = 'yolo-voc.weights'
        print(" YOLOv2 (used pre-train weights) : ", model_weights_file_root)
        weights_loader = WeightLoader()
        weights_loader.load(model, model_weights_file_root, backbone=False)

    # ================================
    # Setting Loss
    # ================================
    criterion = Yolo_Loss(num_classes=opts.num_classes)

    # ================================
    # Setting Optimizer
    # ================================
    optimizer = optim.SGD(params=model.parameters(),
                          lr=opts.lr,
                          momentum=0.9,
                          weight_decay=5e-4)

    # ================================
    # Setting Scheduler
    # ================================
    scheduler = StepLR(optimizer=optimizer, step_size=100, gamma=0.1)

    # ================================
    # Setting Loss
    # ================================

    if opts.start_epoch != 0:
        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
                                .format(opts.start_epoch - 1))           # train
        model.load_state_dict(checkpoint['model_state_dict'])            # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    # load optim state dict
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])    # load sched state dict
        print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))

    else:
        print('\nNo check point to resume.. train from scratch.\n')

    # ================================
    # Training
    # ================================

    for epoch in range(opts.start_epoch, opts.epochs):

        train(epoch=epoch,
              vis=vis,
              train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=scheduler,
              save_path=opts.save_path,
              save_file_name=opts.save_file_name)

        if scheduler is not None:
            scheduler.step()

if __name__ == '__main__':
    main()