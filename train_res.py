# train with backbone resnet50
import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model_resnet import EAST
from loss import Loss
import os
import time
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


def drawLoss(train_loss, save_name):
    x1 = range(0, len(train_loss))

    plt.plot(x1, train_loss, c='red', label='train loss')
    plt.xlabel('item number')
    plt.legend(loc='upper left')
    plt.savefig(save_name, format='jpg')
    plt.close()


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval):
    file_num = len(os.listdir(train_img_path))
    trainset = custom_dataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)

    criterion = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter // 3, epoch_iter * 2 // 3], gamma=0.1)

    train_loss = []

    for epoch in range(epoch_iter):
        model.train()
        scheduler.step()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(
                device)
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                epoch + 1, epoch_iter, i + 1, int(file_num / batch_size), time.time() - start_time, loss.item()))

        epoch_loss_mean = epoch_loss / int(file_num / batch_size)
        train_loss.append(epoch_loss_mean)
        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss_mean,
                                                                  time.time() - epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('=' * 50)
        if (epoch + 1) % interval == 0:
            savePath = pths_path + 'lossImg' + str(epoch + 1) + '.jpg'
            drawLoss(train_loss, savePath)
            lossPath = pths_path + 'loss' + str(epoch + 1) + '.npy'
            train_loss_np = np.array(train_loss, dtype=float)
            np.save(lossPath, train_loss_np)
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch + 1)))
            lr_state = scheduler.state_dict()
            torch.save(lr_state, os.path.join(pths_path, 'scheduler_epoch_{}.pth'.format(epoch + 1)))


if __name__ == '__main__':
    # train_img_path = os.path.abspath('../ICDAR_2015/train_img')
    # train_gt_path  = os.path.abspath('../ICDAR_2015/train_gt')
    train_img_path = '/data/home/zjw/pythonFile/masktextspotter.caffe2/lib/datasets/data/icdar2015/train_images/'
    train_gt_path = '/data/home/zjw/pythonFile/masktextspotter.caffe2/lib/datasets/data/icdar2015/train_gts/'
    pths_path = './pths'
    batch_size = 50
    lr = 1e-3
    num_workers = 8
    epoch_iter = 1000
    save_interval = 20
    train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)

