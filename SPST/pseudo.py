import argparse
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
from dataset.vaihingen import VaihingenDataSet
from dataset.potsdam import PotsdamDataSet
from model.deeplabv2_feature import DeeplabV2_101
from torch.utils import data, model_zoo
from collections import Counter
from scipy.stats import entropy

DATA_DIRECTORY = r'C:\Users\Lenovo\Desktop\no_select'
DATA_LIST_PATH = r'C:\Users\Lenovo\Desktop\no_select\image'
SAVE_PATH = r'C:\Users\Lenovo\Desktop\prob\label'
RESTORE_FROM = r'G:\Vaihingen_Potsdam\model\Potsdam_22000.pth'
NUM_CLASSES = 6


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
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    return parser.parse_args()


def shannon_entropy(feature_map):
    entropy_map = np.zeros((feature_map.shape[0], feature_map.shape[1]), dtype=np.float32)

    for i in range(feature_map.shape[0]):
        for j in range(feature_map.shape[1]):
            # 统计每个值的出现次数
            value_counts = Counter(feature_map[i, j, :])

            shannon_entropy = 0.0
            for count in value_counts.values():
                # 计算每个值的概率
                value_prob = count / feature_map.shape[2]
                # 计算每个值的香农熵贡献
                shannon_entropy -= value_prob * np.log2(value_prob)
            entropy_map[i, j] = shannon_entropy

    return entropy_map


def main():

    args = get_arguments()
    device = torch.device("cuda" if not args.cpu else "cpu")

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = DeeplabV2_101(in_channels=3, n_class=args.num_classes)
    model.load_state_dict(torch.load(args.restore_from, map_location=device))
    model = model.to(device)
    model.eval()

    targetloader = data.DataLoader(PotsdamDataSet(args.data_dir, args.data_list), batch_size=1, shuffle=False, pin_memory=True)

    predictes_label = np.zeros((len(targetloader), 512, 512))
    predictes_prob = np.zeros((len(targetloader), 512, 512))
    predictes_entropy = np.zeros((len(targetloader), 512, 512))
    image_name = []

    interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
    for index, batch in tqdm(enumerate(targetloader), total=len(targetloader)):
        image, label, _, name, _ = batch
        _, predict, _ = model(image.to(device))
        predict = interp(predict)
        predict = nn.functional.softmax(predict, dim=1)

        predict = predict.cpu().data[0].numpy()
        predict = predict.transpose(1, 2, 0)

        predict_label = np.argmax(predict, axis=2)
        predict_prob = np.max(predict, axis=2)
        predict_entropy = entropy(predict, axis=2)
        predictes_label[index] = predict_label.copy()
        predictes_prob[index] = predict_prob.copy()
        predictes_entropy[index] = predict_entropy.copy()

        image_name.append(name[0])

    prob_thres = []
    for i in range(args.num_classes):
        x = predictes_prob[predictes_label == i]
        if len(x) == 0:
            prob_thres.append(0)
            continue
        x = np.sort(x)
        prob_thres.append(x[int(np.round(len(x)*0.25))])
    print(prob_thres)
    prob_thres = np.array(prob_thres)
    prob_thres[prob_thres > 0.9] = 0.9
    print(prob_thres)

    time.sleep(0.001)
    for index in tqdm(range(len(targetloader))):
        name = image_name[index]
        label = predictes_label[index]
        prob = predictes_prob[index]
        for i in range(args.num_classes):
            label[(prob < prob_thres[i]) * (label == i)] = 5
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        output.save('%s/%s' % (args.save, name))
    
    
if __name__ == '__main__':
    main()