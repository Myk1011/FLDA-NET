import pandas as pd
import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from skimage import io, segmentation


#映射
def message_Label_Accuracy(label_path, slic):
    label = io.imread(label_path)

    message = ""
    sps = {}
    rows, cols = label.shape
    max_num = np.max(slic) + 1

    # 256个超像素
    for sp in range(int(max_num)):
        sps[sp] = []

    # 记录各超像素内的label值，按照图像像素遍历
    for i in range(rows):
        for j in range(cols):
            catagory = label[i, j]
            if catagory != 5:
                sp = slic[i, j]  # 获取当前像素的对应的超像素编号---单通道 多通道
                sps[sp].append(catagory)

    # 评价超像素，累计label图像的打分值 以超像素块作为单元循环
    error = 0
    for sp in range(max_num):
        se = pd.Series(sps[sp], dtype='float64')
        countDict = dict(se.value_counts())
        if countDict:
            error += (sum(countDict.values()) - max(countDict.values()))

    # 将打分值记录
    message += str(os.path.basename(label_path)) + ": " + str(error) + '\n'
    print(message)

    return error, message


def trans_Slic(imgloc="", labloc="", slicloc="", scale="", sigma="", min_size=""):

    dir_name = f"{scale}_{sigma}_{min_size}"
    dir_loc = os.path.join(slicloc, dir_name)
    if not os.path.exists(dir_loc):
        os.makedirs(dir_loc)

    imglst = os.listdir(imgloc)
    lbllst = os.listdir(labloc)

    picnum = len(imglst)
    mark = {}
    messages = ""
    for i in tqdm(range(picnum)):
        image = io.imread(os.path.join(imgloc, imglst[i]))
        segment = segmentation.felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
        print(imglst[i], np.max(segment))
        cv2.imwrite(os.path.join(dir_loc, imglst[i]), segment)
        error, message = message_Label_Accuracy(label_path=os.path.join(labloc, lbllst[i]), slic=segment)
        mark[imglst[i]] = error
        messages += message

    sort = sorted(mark.items(), key=lambda x: x[1])
    print(sort)
    messages += str(sort)

    with open(os.path.join(dir_loc + '/' + str(scale) + '_' + str(sigma) + '_' + str(min_size) + '_'
                           + 'result.txt'), 'w') as mark_txt:
        mark_txt.write(messages)


if __name__ == '__main__':
    imgloc = r"C:\Users\Lenovo\Desktop\毕业\孟雨柯\小论文\模型对比\Vaihingen_Potsdam\pseudo\output\image"
    labelloc = r"C:\Users\Lenovo\Desktop\毕业\孟雨柯\小论文\模型对比\Vaihingen_Potsdam\pseudo\output\predict"
    slicloc = r"C:\Users\Lenovo\Desktop\slic"
    trans_Slic(imgloc=imgloc, labloc=labelloc, slicloc=slicloc, scale=1, sigma=0.8, min_size=10)