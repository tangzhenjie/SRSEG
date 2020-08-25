import argparse
import os
import sys
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter

from data import GenerateData
from SRSEGNet import SRSEGNet

from loss import FALoss

from metrics import RunningScore

def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for SRSEG network")
    main_parser.add_argument("--cuda", type=int, required=False, default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    main_parser.add_argument("--batch_size", type=int, default=12, help="batch size, default set to 64")
    main_parser.add_argument("--epochs", type=int, default=60, help="epochs, default set to 20")
    main_parser.add_argument("--num_classes", type=int, default=2, help="the class for segmentation")
    # datasets
    main_parser.add_argument("--dataset_name", type=str, default="InriaDataset",
                              help="dataset_name, default set to dataset_name")
    main_parser.add_argument("--img_size", type=int, default=256, help="the img size to crop for training")
    main_parser.add_argument("--up_scale", type=int, default=2, help="the img size to super resolution")
    # training
    main_parser.add_argument("--model_title", type=str, default="SRSEGNet",
                              help="model_title, default set to model_title")
    main_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    main_parser.add_argument("--learning_rate", type=float, default=0.01,
                              help="learning rate, default set to 1e-4")
    main_parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay, default set to 0")
    main_parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 7)")

    args = main_parser.parse_args()
    print(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    train(args)


# global settings
resume = False
log_interval = 50

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    print('===> Loading datasets')
    train_path = './datasets/' + args.dataset_name + '/trains/'
    test_path  = './datasets/' + args.dataset_name + '/tests/'

    train_set = GenerateData(image_dir=train_path, img_size=args.img_size, lr_img_size=args.img_size // args.up_scale, augment=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)


    print('===> Building model')
    net = SRSEGNet(num_classes=args.num_classes, up_scale=int(8 * args.up_scale))
    #print(net)
    model_title = args.dataset_name + "_" + args.model_title + '_imgsize=' + str(args.img_size) + '_upscale' + str(args.up_scale)
    model_name = './checkpoints/' + model_title + "_ckpt_epoch_" + str(50) + ".pth"
    args.model_title = model_title
    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)
    start_epoch = 0
    if resume:
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(model_name))
    net.to(device).train()

    # loss functions to choose
    FA_loss = FALoss()
    SR_loss = torch.nn.MSELoss()
    SEG_loss = torch.nn.CrossEntropyLoss()

    print("===> Setting optimizer and logger")
    # add L2 regularization
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epoch_meter = meter.AverageValueMeter()
    writer = SummaryWriter('./runs/' + model_title + '_' + str(time.ctime())) #  + str(time.ctime())


    print('===> Start training')
    for e in range(start_epoch, args.epochs):
        epoch_meter.reset()
        print("Start epoch {}, learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
        for iteration, (img, label, img_lr) in enumerate(train_loader):
            # adjust the learning rate   e * len(train_loader) + iteration + 1
            adjust_learning_rate(args.learning_rate, optimizer, e * len(train_loader) + iteration + 1,
                                 args.epochs * len(train_loader), power=0.9)
            img, label, img_lr = img.to(device), label.to(device), img_lr.to(device)
            optimizer.zero_grad()
            img_sr, seg_pre, feature_sr, feature_seg = net(img_lr)
            seg = seg_pre.data.max(1)[1]     # for visualization
            loss_sr = SR_loss(img_sr, img)
            loss_seg = SEG_loss(seg_pre, label.long().squeeze(1))
            loss_FA = FA_loss(feature_sr, feature_seg)
            loss_all = 0.1 * loss_sr + loss_seg + loss_FA
            epoch_meter.add(loss_all.item())
            loss_all.backward()
            optimizer.step()

            # tensorboard visualization
            if (iteration + log_interval) % log_interval == 0:
                print("===> {} GPU{}\tEpoch[{}]({}/{}): Loss: {:.6f}".format(time.ctime(),
                                                                             args.gpus, e + 1,
                                                                             iteration + 1,
                                                                             len(train_loader), loss_all.item()))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss_all, n_iter)
            if iteration == 0:
                writer.add_image('image/epoch' + str(e) + 'img', (img.cpu().numpy()[0, :, :, :] + 1) / 2.0)
                writer.add_image('image/epoch' + str(e) + 'label', label.cpu().numpy()[0, :, :, :])
                writer.add_image('image/epoch' + str(e) + 'img_lr', (img_lr.cpu().numpy()[0, :, :, :] + 1) / 2.0)
                writer.add_image('image/epoch' + str(e) + 'img_sr', (img_sr.cpu().detach().numpy()[0, :, :, :] + 1) / 2.0)
                writer.add_image('image/epoch' + str(e) + 'seg', seg.unsqueeze(1).cpu().numpy()[0, :, :, :])
        print("===> {}\tEpoch {} Training Complete: Avg. Loss: {:.6f}".format(time.ctime(), e + 1,
                                                                              epoch_meter.value()[0]))
        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss', epoch_meter.value()[0], e + 1)

        # save model weights at checkpoints every 10 epochs
        if (e + 1) % 10 == 0:
            save_checkpoint(args, net, e + 1)
    ## Save the testing results
    print("Running testset")
    print('===> Loading testset')
    test_set = GenerateData(image_dir=test_path, img_size=512, lr_img_size=512 // args.up_scale, augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    net.eval().cuda()
    with torch.no_grad():
        metrics = RunningScore(args.num_classes)
        for i, (img, label, img_lr) in enumerate(test_loader):
            img, label, img_lr = img, label, img_lr.to(device)
            img_sr, seg_pre, feature_sr, feature_seg = net(img_lr)
            seg = seg_pre.data.max(1)[1]  # for visualization
            if i % 20 == 0:
                writer.add_image('image/epoch' + str(i) + 'img', (img.numpy()[0, :, :, :] + 1) / 2.0)
                writer.add_image('image/epoch' + str(i) + 'label', label.numpy()[0, :, :, :])
                writer.add_image('image/epoch' + str(i) + 'img_lr', (img_lr.cpu().numpy()[0, :, :, :] + 1) / 2.0)
                writer.add_image('image/epoch' + str(i) + 'img_sr', (img_sr.cpu().detach().numpy()[0, :, :, :] + 1) / 2.0)
                writer.add_image('image/epoch' + str(i) + 'seg', seg.unsqueeze(1).cpu().numpy()[0, :, :, :])
            gt = np.squeeze(label.numpy(), axis=1)
            pre = seg.cpu().numpy()
            metrics.update(gt, pre)
        val_class_iou, iu = metrics.get_scores()
        print("IOU:" + str(val_class_iou))

        # poly learning rate
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))
def adjust_learning_rate(start_lr, optimizer, i_iter, max_iter, power=0.9):
    lr = lr_poly(start_lr, i_iter, max_iter, power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def save_checkpoint(args, model, epoch):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    checkpoint_model_dir = './checkpoints/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.model_title + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    state = {"epoch": epoch, "model": model}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))
if __name__ == "__main__":
    main()
