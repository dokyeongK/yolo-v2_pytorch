from torch.utils.data import Dataset
import os
import glob

class VOC_Dataset(Dataset):
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root='/mnt/mydisk/hdisk/dataset/VOCdevkit', mode='TRAIN', year = 'VOC2007'):
        super(VOC_Dataset, self).__init__()

        root = os.path.join(root, mode, year)

        # VOC data image, annotation list 불러오
        self.img_list = sorted(glob.glob(os.path.join(root, 'JPEGImages/*.jpg')))
        self.anno_list = sorted(glob.glob(os.path.join(root, 'Annotations/*.xml')))
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.class_dict_inv = {i: class_name for i, class_name in enumerate(self.class_names)}

        print(len(self.anno_list))

    def __len__(self):
        return self.img_list

if __name__ == '__main__':
    train_dataset = VOC_Dataset(root='/mnt/mydisk/hdisk/dataset/VOCdevkit', mode='TRAIN', year='VOC2007')