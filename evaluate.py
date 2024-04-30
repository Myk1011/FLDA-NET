import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data, model_zoo
from dataset.vaihingen import VaihingenDataSet
from dataset.potsdam import PotsdamDataSet
from utils.metric import Evaluator
from PIL import Image
from model.deeplabv2_feature import DeeplabV2_101
from prettytable import PrettyTable
from tqdm import tqdm

NUM_CLASSES = 6
DATA_DIRECTORY = r'G:\P_V_resample\cyclegan\likeV\train\Vaihingen'
DATA_LIST_PATH = r'G:\P_V_resample\cyclegan\likeV\train\Vaihingen\image'
RESTORE_FROM = r'G:\Potsdam_Vaihingen\Full_level\model\Potsdam_22000.pth'
SAVE_PATH = r'C:\Users\Administrator\Desktop\output'
CLASSES = ('ImSurf', 'Tree', 'Building', 'Car', 'LowVeg', 'Clutter')


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 255, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [255, 0, 0]

    return mask_rgb


def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = DeeplabV2_101(in_channels=3, n_class=args.num_classes)
    model.load_state_dict(torch.load(args.restore_from, map_location=torch.device('cuda')))
    model.cuda(gpu0)
    model.eval()
    
    testloader = data.DataLoader(VaihingenDataSet(args.data_dir, args.data_list), batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    evaluator = Evaluator(num_class=args.num_classes, log_name='vaihingen')
    evaluator.reset()

    with torch.no_grad():
        for index, batch in tqdm(enumerate(testloader), total=len(testloader)):
            transform_image, label, _, name, image = batch

            output1, output2, feature = model(Variable(transform_image).cuda(gpu0))
            predict = interp(output1).cpu().data[0].numpy()
            predict = predict.transpose(1, 2, 0)
            predict = np.asarray(np.argmax(predict, axis=2), dtype=np.uint8)

            name = name[0].split('/')[-1]

            label = label[0].cpu().numpy()
            evaluator.add_batch(label, predict)

            # 输出结果图片
            label = label2rgb(label)
            predict = label2rgb(predict)
            image = Image.fromarray(image[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
            label = Image.fromarray(label)
            predict = Image.fromarray(predict)
            image.save('%s/image/%s' % (args.save, name))
            label.save('%s/label/%s' % (args.save, name))
            predict.save('%s/predict/%s' % (args.save, name))

        test_m_IoU = np.nanmean(evaluator.Intersection_over_Union()[:-1])
        test_m_F1 = np.nanmean(evaluator.F1()[:-1])
        test_recall = np.nanmean(evaluator.Recall()[:-1])
        test_precision = np.nanmean(evaluator.Precision()[:-1])
        test_iou_per_class = evaluator.Intersection_over_Union()
        test_F1_per_class = evaluator.F1()
        test_OA = evaluator.OA()
        test_Kappa = evaluator.Kappa()

        test_table = PrettyTable(["序号", "名称", "F1", "IoU"])
        for i in range(args.num_classes):
            test_table.add_row([i, CLASSES[i], test_F1_per_class[i], test_iou_per_class[i]])
        print(test_table)

        print('\n'f'test_OA:{test_OA}, test_F1_score:{test_m_F1}, test_mIoU:{test_m_IoU}')
        print(f'test_recall:{test_recall}, test_precision:{test_precision}, test_kappa:{test_Kappa}')


if __name__ == '__main__':
    main()