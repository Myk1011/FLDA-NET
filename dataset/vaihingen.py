import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms

transform_V_resample = transforms.Normalize(mean=[118.7296385992141, 81.12362460181826, 80.03289021083287],
                                             std=[54.82199391515325, 39.1127696407364, 37.351839566769385])

# transform_likeP_resample = transforms.Normalize(mean=[86.10290778023857, 90.53763780139741, 85.62965552920387],
#                                                  std=[35.569135027641074, 37.08393100481927, 39.19142479993313])


class VaihingenDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None):
        self.root = root
        self.list_path = list_path
        self.img_ids = os.listdir(list_path)
        print(f'Vaihingen: {len(self.img_ids)}')
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
        transform_image = transform_V_resample(image)

        size = image.shape

        return transform_image, label, np.array(size), name, image