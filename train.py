import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import numpy as np
from torch.utils import data
from torch.autograd import Variable
from model.deeplabv2_feature import DeeplabV2_101
from model.discriminator import FCDiscriminator
from model.discriminator_feature import FCDiscriminator_feature
from losses.loss import CrossEntropy2d
from losses.loss import WeightedBCEWithLogitsLoss
from losses.joint_loss import JointLoss
from losses.dice import DiceLoss
from losses.soft_ce import SoftCrossEntropyLoss
from dataset.potsdam import PotsdamDataSet
from dataset.vaihingen import VaihingenDataSet

BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
IGNORE_LABEL = 5
MOMENTUM = 0.9
POWER = 0.9
NUM_CLASSES = 6
SAVE_PRED_EVERY = 2000
RESTORE_FROM = None

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS_STOP/20)
Lambda_adv_o = 0.01
Lambda_adv_f = 0.001
Lambda_local = 40
Epsilon = 1.0

INPUT_SIZE_SOURCE = '512, 512'
INPUT_SIZE_TARGET = '512, 512'
DATA_DIRECTORY = r'E:\P_V_resample\cyclegan\likeV\train\Potsdam'
DATA_LIST_PATH = r'E:\P_V_resample\cyclegan\likeV\train\Potsdam\image'
DATA_DIRECTORY_TARGET = r'\P_V_resample\cyclegan\likeV\train\Vaihingen'
DATA_LIST_PATH_TARGET = r'\P_V_resample\cyclegan\likeV\train\Vaihingen\image'
SNAPSHOT_DIR = r'E:\pycharmwork\FLDA-NET\P_V/'


def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size-source", type=str, default=INPUT_SIZE_SOURCE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate_D, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_feature(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def weightmap(pred1, pred2):
    output = 1.0 - torch.sum((pred1 * pred2), dim=1).view(1, BATCH_SIZE, pred1.size(2), pred1.size(3)) / \
    (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(1, BATCH_SIZE, pred1.size(2), pred1.size(3))
    return output


def main():
    """Create the model and start the training."""
    h, w = map(int, args.input_size_source.split(','))
    input_size_source = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    cudnn.enabled = True
    
    # Create Network
    model = DeeplabV2_101(in_channels=3, n_class=args.num_classes)
    if args.restore_from is not None:
        model.load_state_dict(torch.load(args.restore_from + '.pth', map_location=torch.device('cuda')))
        
    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # Init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    model_feature = FCDiscriminator_feature(num_classes=args.num_classes)

# =============================================================================
#    #for retrain
#    saved_state_dict_D = torch.load(RESTORE_FROM_D)
#    model_D.load_state_dict(saved_state_dict_D)
# =============================================================================

    model_D.train()
    model_D.cuda(args.gpu)
    model_feature.train()
    model_feature.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    trainloader = data.DataLoader(PotsdamDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(VaihingenDataSet(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.iter_size * args.batch_size),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    optimizer_feature = optim.Adam(model_feature.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_feature.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()
    seg_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=args.ignore_label),
                         DiceLoss(mode='multiclass', smooth=0.05, ignore_index=args.ignore_label), 1.0, 1.0)

    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
    
    # Labels for Adversarial Training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        optimizer_feature.zero_grad()
        adjust_learning_rate_feature(optimizer_feature, i_iter)
        
        damping = (1 - i_iter/NUM_STEPS)

        #======================================================================================
        # train G
        #======================================================================================

        #Remove Grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        for param in model_feature.parameters():
            param.requires_grad = False

        # Train with Source
        _, batch = next(trainloader_iter)
        images_s, labels_s, _, _, _ = batch
        images_s = Variable(images_s).cuda(args.gpu)
        labels_s = labels_s.long().cuda(args.gpu)
        pred_source1, pred_source2, feature_source = model(images_s)
        pred_source1 = interp_source(pred_source1)
        pred_source2 = interp_source(pred_source2)
		
        #Segmentation Loss
        loss_seg = seg_loss(pred_source2, labels_s)
        loss_seg.backward()

        # Train with Target
        _, batch = next(targetloader_iter)
        images_t, _, _, _, _ = batch
        images_t = Variable(images_t).cuda(args.gpu)
        pred_target1, pred_target2, feature_target = model(images_t)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)

        weight_map = weightmap(F.softmax(pred_target1, dim=1), F.softmax(pred_target2, dim=1))

        D_out = interp_target(model_D(F.softmax(pred_target2, dim=1)))
        D_feature = model_feature(prob_2_entropy(F.softmax(feature_target, dim=1)))

        #Adaptive Adversarial Loss
        if(i_iter > PREHEAT_STEPS):
            loss_adv = weighted_bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(args.gpu), weight_map, Epsilon, Lambda_local)
        else:
            loss_adv = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(args.gpu))

        loss_adv_feature = bce_loss(D_feature, Variable(torch.FloatTensor(D_feature.data.size()).fill_(source_label)).cuda(args.gpu))

        loss = loss_adv * Lambda_adv_o * damping + loss_adv_feature * Lambda_adv_f
        loss.backward()
        
        #======================================================================================
        # train D
        #======================================================================================
        
        # Bring back Grads in D
        for param in model_D.parameters():
            param.requires_grad = True

        for param in model_feature.parameters():
            param.requires_grad = True

        # Train with Source
        pred_source1 = pred_source1.detach()
        pred_source2 = pred_source2.detach()
        feature_source = feature_source.detach()

        D_out_s = interp_source(model_D(F.softmax(pred_source2, dim=1)))
        D_feature_s = model_feature(prob_2_entropy(F.softmax(feature_source, dim=1)))

        loss_D_s = bce_loss(D_out_s, Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda(args.gpu))
        loss_D_s.backward()

        loss_D_s_feature = bce_loss(D_feature_s, Variable(torch.FloatTensor(D_feature_s.data.size()).fill_(source_label)).cuda(args.gpu))
        loss_D_s_feature.backward()

        # Train with Target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()
        feature_target = feature_target.detach()
        weight_map = weight_map.detach()

        D_out_t = interp_target(model_D(F.softmax(pred_target2, dim=1)))
        D_feature_t = model_feature(prob_2_entropy(F.softmax(feature_target, dim=1)))

        #Adaptive Adversarial Loss
        if(i_iter > PREHEAT_STEPS):
            loss_D_t = weighted_bce_loss(D_out_t, Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(args.gpu), weight_map, Epsilon, Lambda_local)
        else:
            loss_D_t = bce_loss(D_out_t, Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(args.gpu))
            
        loss_D_t.backward()

        loss_D_t_feature = bce_loss(D_feature_t, Variable(torch.FloatTensor(D_feature_t.data.size()).fill_(target_label)).cuda(args.gpu))
        loss_D_t_feature.backward()

        optimizer.step()
        optimizer_D.step()
        optimizer_feature.step()

        print('exp = {}'.format(args.snapshot_dir))
        print('iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_adv = {3:.4f}, loss_adv_feature = {4:.4f}, loss_D_s = {5:.4f} loss_D_t = {6:.4f}'
              .format(i_iter, args.num_steps, loss_seg, loss_adv, loss_adv_feature, loss_D_s, loss_D_t))

        f_loss = open(osp.join(args.snapshot_dir, 'loss.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n'.format(loss_seg, loss_adv, loss_adv_feature, loss_D_s, loss_D_t))
        f_loss.close()
        
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'Potsdam_' + str(args.num_steps) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'Potsdam_' + str(args.num_steps) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'Potsdam_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'Potsdam_' + str(i_iter) + '_D.pth'))


if __name__ == '__main__':
    main()