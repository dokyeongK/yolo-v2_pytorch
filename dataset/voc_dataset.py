from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import torch
import numpy as np
from xml.etree.ElementTree import parse
import transform
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
class VOC_Dataset(Dataset):


    def __init__(self, root='/mnt/mydisk/hdisk/dataset/VOCdevkit', mode='TRAIN', year = 'VOC2007'):
        super(VOC_Dataset, self).__init__()

        root = os.path.join(root, mode, year)

        # VOC data image, annotation list 불러오기
        self.img_list = sorted(glob.glob(os.path.join(root, 'JPEGImages/*.jpg')))
        self.anno_list = sorted(glob.glob(os.path.join(root, 'Annotations/*.xml')))

        self.class_names = ('aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.class_dict_inv = {i: class_name for i, class_name in enumerate(self.class_names)}
        self.img_size = 416
        self.mode = mode

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        visualize = False

        img = Image.open(self.img_list[idx]).convert('RGB')

        # is_difficult : 0 pr 1
        boxes, labels, is_difficult = self.parse_voc(self.anno_list[idx])

        # load img name for string
        img_name = os.path.basename(self.anno_list[idx]).split('.')[0]

        # image name -> ascii
        img_name_to_ascii = [ord(c) for c in img_name]

        img_width, img_height = float(img.size[0]), float(img.size[1])

        # convert to tensor
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        difficulties = torch.ByteTensor(is_difficult)  # (n_objects)
        img_name = torch.FloatTensor([img_name_to_ascii])
        additional_info = torch.FloatTensor([img_width, img_height])

        image, boxes, labels, difficulties = transform.transform_voc(img, boxes, labels, difficulties, self.mode)

        if visualize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # tensor to img
            img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H

            img_vis *= std
            img_vis += mean

            img_vis = np.clip(img_vis, 0, 1)

            plt.figure('img')
            plt.imshow(img_vis)
            print('num objects : {}'.format(len(boxes)))
            for i in range(len(boxes)):
                print(boxes[i], labels[i])
                plt.gca().add_patch(Rectangle((boxes[i][0] * self.img_size, boxes[i][1] * self.img_size),
                                              boxes[i][2] * self.img_size - boxes[i][0] * self.img_size,
                                              boxes[i][3] * self.img_size - boxes[i][1] * self.img_size,
                                              linewidth=1, edgecolor='r', facecolor='none'))
                plt.text(boxes[i][0] * self.img_size - 10, boxes[i][1] * self.img_size - 10,
                         str(self.class_dict_inv[labels[i].item()]),
                         bbox=dict(boxstyle='round4', color='grey'))

            plt.show()
        if self.mode == "TEST":
            return image, boxes, labels, difficulties, img_name, additional_info  # for evaluations

        return image, boxes, labels, difficulties


    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        img_name = list()
        difficulties = list()

        if self.mode == "TEST":
            additional_info = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])
            if self.mode == "TEST":
                img_name.append(b[4])
                additional_info.append(b[5])

        images = torch.stack(images, dim=0)
        if self.mode == "TEST":
            return images, boxes, labels, difficulties, img_name, additional_info
        return images, boxes, labels, difficulties

    def parse_voc(self, xml_file_path):

        tree = parse(xml_file_path)
        root = tree.getroot()

        boxes = []
        labels = []
        is_difficult = []
# -------------------------------------------------------------------------------------------------------------------------------------
        for obj in root.iter("object"):

            # <name>
            name = obj.find('./name')

            # <name> -> 소문자로
            class_name = name.text.lower().strip()

            # class name -> label (ex, chair -> 8)
            labels.append(self.class_dict[class_name])

            # stop to bbox tag
            bbox = obj.find('./bndbox')
            x_min = bbox.find('./xmin')
            y_min = bbox.find('./ymin')
            x_max = bbox.find('./xmax')
            y_max = bbox.find('./ymax')

            # from str to int # 왜 1을 빼지? question 1
            x_min = float(x_min.text) - 1
            y_min = float(y_min.text) - 1
            x_max = float(x_max.text) - 1
            y_max = float(y_max.text) - 1

            boxes.append([x_min, y_min, x_max, y_max])
            # is_difficult
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

# if __name__ == '__main__':
#     train_dataset = VOC_Dataset(root='/mnt/mydisk/hdisk/dataset/VOCdevkit', mode='TRAIN', year='VOC2007')
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=1,
#                                                collate_fn=train_dataset.collate_fn,
#                                                shuffle=False,
#                                                num_workers=0,
#                                                pin_memory=True)
#
#     for i, (images, boxes, labels, _) in enumerate(train_loader):
#         images = images.cuda()
#         boxes = [b.cuda() for b in boxes]
#         labels = [l.cuda() for l in labels]
#
#     print("end")
