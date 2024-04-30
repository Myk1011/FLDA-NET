import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms

transform_likeV_resample = transforms.Normalize(mean=[115.04210585890954, 81.69292352354981, 77.29196238312481],
                                                 std=[52.70807977648717, 37.07916402625995, 34.252774026644545])

# transform_P_resample = transforms.Normalize(mean=[85.7993591843496, 91.65151117472631, 84.83360369824248],
#                                              std=[35.53222167578009, 34.863165535781874, 36.28596868560013])


class PotsdamDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None):
        self.root = root
        self.list_path = list_path
        self.img_ids = os.listdir(list_path)
        print(f'Potsdam: {len(self.img_ids)}')
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, "image/%s" % name)
            label_file = os.path.join(self.root, "label/%s" % name)
            self.files.append({"img": img_file, "label": label_file, "name": name})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        transform_image = transform_likeV_resample(image)

        size = image.shape

        return transform_image, label, np.array(size), name, image