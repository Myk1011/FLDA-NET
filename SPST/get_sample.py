from PIL import Image
import os
import numpy as np
from tqdm import tqdm


def openreadtxt(path):
    data = []
    file = open(path, 'r')  #打开文件
    file_data = file.readlines() #读取所有行
    for row in file_data:
        tmp_list = row.split(' ') #按‘，'切分每行的数据
        #tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
        data.append(tmp_list) #将每行数据插入data中
    return data


def extract_txt_name(txt_path, select_rate):
    images_name = []
    txt = openreadtxt(txt_path)
    for index, dict_name in enumerate(txt[len(txt)-1][:int(select_rate * len(txt[len(txt)-1]))]):
    # for index, dict_name in enumerate(txt[len(txt)-1]):
        if index % 2 == 0:
            name = dict_name.split('\'')[1]
            images_name.append(name)
    return images_name


if __name__ == '__main__':
    image_path = r'G:\Potsdam_Vaihingen\pseudo\no_select\image'
    pseudo_path = r'G:\Potsdam_Vaihingen\pseudo\no_select\label'
    txt_path = r'G:\Potsdam_Vaihingen\pseudo\slic_acc\10_10_10\10_10_10_result.txt'
    save_path = r'G:\Potsdam_Vaihingen\pseudo\slic_acc\50'

    images_name = extract_txt_name(txt_path, select_rate=0.5)
    for name in tqdm(images_name):
        image = Image.open(os.path.join(image_path, name))
        image.save(os.path.join(save_path + '\image', name))
        label = Image.open(os.path.join(pseudo_path, name))
        label.save(os.path.join(save_path + '\label', name))