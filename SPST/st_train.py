from torch.utils.data import DataLoader
from tqdm import tqdm
from time import perf_counter, sleep
from model.deeplabv2_feature import DeeplabV2_101
from losses.joint_loss import JointLoss
from losses.dice import DiceLoss
from losses.soft_ce import SoftCrossEntropyLoss
from dataset.potsdam import PotsdamDataSet
from dataset.vaihingen import VaihingenDataSet
from catalyst.contrib.nn import Lookahead
from utils.metric import Evaluator
from prettytable import PrettyTable
from pytorch_lightning import seed_everything
import torch.nn.functional as F
import numpy as np
import torch
import os


def model_train(trainLoader, testLoader, model, criterion, optimizer, scheduler,
                num_classes, epoch_num, save_path, classes):
    interp = torch.nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    best_val_F1 = 0
    for epoch in range(1, epoch_num + 1):
        with tqdm(total=len(trainLoader), desc=f'Epoch {epoch}/{epoch_num} ', unit='batch') as pbar:
            #  训练
            metrics_train = Evaluator(num_class=num_classes, log_name='vaihingen')
            metrics_train.reset()  # 重置混淆矩阵
            model.train()
            for progress, data in enumerate(trainLoader):
                image, label = data[0].to(device), data[1].to(device)
                _, predict, _ = model(image)
                predict = interp(predict)
                loss = criterion(predict, label.long())
                loss.backward()
                predict = torch.max(predict, dim=1)[1]
                for i in range(label.shape[0]):
                    metrics_train.add_batch(label[i].cpu().numpy(), predict[i].cpu().numpy())
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step(loss.item())
                train_OA = metrics_train.OA()
                pbar.set_postfix({f'loss': loss.detach(), f'train_OA': train_OA, f'lr': optimizer.param_groups[0]["lr"]})
                pbar.update(1)

            #  计算指标
            train_m_IoU = np.nanmean(metrics_train.Intersection_over_Union()[:-1])
            train_m_F1 = np.nanmean(metrics_train.F1()[:-1])
            train_OA = metrics_train.OA()
            train_F1_per_class = metrics_train.F1()
            train_iou_per_class = metrics_train.Intersection_over_Union()
            train_Kappa = metrics_train.Kappa()
            train_table = PrettyTable(["Number", "Name", "F1", "IoU"])
            for i in range(num_classes):
                train_table.add_row([i, classes[i], train_F1_per_class[i], train_iou_per_class[i]])
            train_log_dict = {'train_mIoU': round(train_m_IoU, 6), 'train_F1': round(train_m_F1, 6),
                              'train_OA': round(train_OA, 6), 'train_Kappa': round(train_Kappa, 6)}
            sleep(0.1)
            print('\n', train_table)
            print(train_log_dict)

            #  验证
            metrics_val = Evaluator(num_class=num_classes, log_name='vaihingen')
            metrics_val.reset()  # 重置混淆矩阵
            model.eval()
            with torch.no_grad():
                for progress, data in enumerate(testLoader):
                    image, label = data[0].to(device), data[1].to(device)
                    _, predict, _ = model(image)
                    predict = interp(predict)
                    predict = torch.max(predict, dim=1)[1]
                    for i in range(label.shape[0]):
                        metrics_val.add_batch(label[i].cpu().numpy(), predict[i].cpu().numpy())

                #  计算指标
                val_m_IoU = np.nanmean(metrics_val.Intersection_over_Union()[:-1])
                val_m_F1 = np.nanmean(metrics_val.F1()[:-1])
                val_OA = metrics_val.OA()
                val_F1_per_class = metrics_val.F1()
                val_iou_per_class = metrics_val.Intersection_over_Union()
                val_Kappa = metrics_val.Kappa()
                val_table = PrettyTable(["Number", "Name", "F1", "IoU"])
                for i in range(num_classes):
                    val_table.add_row([i, classes[i], val_F1_per_class[i], val_iou_per_class[i]])
                val_log_dict = {'val_mIoU': round(val_m_IoU, 6), 'val_F1': round(val_m_F1, 6),
                                'val_OA': round(val_OA, 6), 'val_Kappa': round(val_Kappa, 6)}
                print(val_table)
                print(val_log_dict)

                #  保存模型
                sleep(0.001)
                name = 'epoch_%d_acc_%.5f_f1_score_%.5f' % (epoch, val_log_dict['val_OA'], val_log_dict['val_F1'])
                torch.save(model.state_dict(), os.path.join(save_path, name + '.pth'))
                if val_log_dict['val_F1'] > best_val_F1:
                    torch.save(model.state_dict(),
                               os.path.join(save_path, 'best_val_F1.pth'))
                    best_val_F1 = val_log_dict['val_F1']
                    print(f'best_val_F1模型已保存: best_val_F1.pth')


if __name__ == '__main__':
    in_channel = 3
    num_classes = 6
    ignore_index = 5
    train_batch_size = 8
    val_batch_size = 8
    numWorkers = 4
    learning_rate = 6e-4
    weight_decay = 0.01
    epoch_num = 100
    seed_everything(77)
    classes = ('ImSurf', 'Tree', 'Building', 'Car', 'LowVeg', 'Clutter')

    #  更换路径
    resume_path = r'G:\Vaihingen_Potsdam\model\Potsdam_22000.pth'
    save_path = r'C:\Users\Lenovo\Desktop\prob\model'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define the model
    model = DeeplabV2_101(in_channel, num_classes).to(device)
    if resume_path is not None:
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print('model.load_state_dict:', resume_path)

    # define the loss
    loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                     DiceLoss(mode='multiclass', smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

    # define the dataloader
    train_dataset = VaihingenDataSet(root=r'C:\Users\Lenovo\Desktop\prob',
                                   list_path=r'C:\Users\Lenovo\Desktop\prob\image')
    test_dataset = VaihingenDataSet(root=r'E:\P_V_resample\cyclegan\likeV\test\Vaihingen',
                                  list_path=r'E:\P_V_resample\cyclegan\likeV\test\Vaihingen\image')
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=4,
                              pin_memory=True, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=val_batch_size, num_workers=4,
                             shuffle=False, pin_memory=True, drop_last=False)

    base_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = Lookahead(base_optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    start = perf_counter()
    sleep(0.001)
    model_train(trainLoader=train_loader,
                testLoader=test_loader,
                model=model,
                criterion=loss,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                num_classes=num_classes,
                epoch_num=epoch_num,
                save_path=save_path,
                classes=classes)
    finish = perf_counter()
    time = finish - start
    print('训练时间：%s', time)