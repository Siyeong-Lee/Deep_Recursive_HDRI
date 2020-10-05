from os import listdir
from os.path import join

from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import random

class data_loader(Dataset):
    def __init__(self, dataset_path, img_size=64, fliplr=True, fliptb=False, rotate=False, gray=False):
        super(data_loader, self).__init__()

        self.img_size = img_size
        self.dataset_path = dataset_path
        self.fliptb = fliptb
        self.fliplr = fliplr
        self.rotate = rotate
        self.gray = gray
        
        self.input_img = [join(dataset_path + '/X/', x) for x in sorted(listdir(dataset_path + '/X/')) if is_image_file(x)]
        self.target_img = [join(dataset_path + '/y/', y) for y in sorted(listdir(dataset_path + '/y/')) if is_image_file(y)]

        assert len(self.input_img) == len(self.target_img)

    def __getitem__(self, index):

        input_img = load_img(self.input_img[index])
        target_img = load_img(self.target_img[index])
        
        if self.rotate:
            rv = random.randint(1,3)
            input_img = input_img.rotate(90 * rv, expand = True)
            target_img = target_img.rotate(90 * rv, expand = True)
 
        if self.fliplr:
            if random.random() < 0.5:
                input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
                target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)
 
        if self.fliptb:
            if random.random() < 0.5:
                input_img = input_img.transpose(Image.FLIP_TOP_BOTTOM)
                target_img = target_img.transpose(Image.FLIP_TOP_BOTTOM)

        total = transforms.Compose([transforms.Scale(4*self.img_size),
                                   transforms.ToTensor()#,
                                   #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   #                     std=[0.229, 0.224, 0.225])
                                   ])

        input_tensor = total(input_img)
        target_tensor = total(target_img)

        return input_tensor, target_tensor

    def __len__(self):
        return len(self.input_img)

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    
    return img

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.PNG', '.JPG'])


def get_loader(image_path, img_size=64, is_gray=False):
    dataset = data_loader(dataset_path=image_path, img_size=img_size)
    return dataset
