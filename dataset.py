import torch
from torch.utils.data.dataset import Dataset
import os
import shutil
import cv2
import numpy as np
import math

from augmentation import data_augmentation
from utils import gaussian_radius, draw_gaussian, resize_image, normalize_image, unnormalize_image

class CenterNetDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, train,
                 input_shape=(512, 512), stride=4, num_augment=2, num_classes=1,
                 ):
        super(CenterNetDataset, self).__init__()
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.train = train

        self.input_shape = input_shape  # w, h
        self.output_shape = (input_shape[0] // stride, input_shape[1] // stride)
        self.stride = stride
        self.num_augment = num_augment
        self.num_classes = num_classes

        self.red_list, self.blue_list, self.green_list = [], [], []
        self.data = []
        self.classes = [0] * num_classes

        self.mean = [0.40789655, 0.44719303, 0.47026116]
        self.std = [0.2886383, 0.27408165, 0.27809834]

        self.data = self.read_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # Initial labels, hm, wh, offset, mask
        batch_labels = np.zeros(self.num_classes, dtype=np.float32)
        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_wh = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_offset = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_offset_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)

        # Get image, annotation
        origin_image, origin_bboxes = self.data[item]

        # Preprocess
        if self.train:
            image, bboxes = data_augmentation(origin_image, self.input_shape, origin_bboxes)
        else:
            image, bboxes = resize_image(origin_image, self.input_shape, origin_bboxes)
        image = np.transpose(normalize_image(image), (2, 0, 1))

        # Get true hm, wh, offset, mask
        for bbox in bboxes:
            label, x, y, w, h = bbox
            label = int(label)
            x1 = int(np.clip((x - w / 2) * self.output_shape[0], 0, self.output_shape[0]))
            x2 = int(np.clip((x + w / 2) * self.output_shape[0], 0, self.output_shape[0]))
            y1 = int(np.clip((y - h / 2) * self.output_shape[1], 0, self.output_shape[1]))
            y2 = int(np.clip((y + h / 2) * self.output_shape[1], 0, self.output_shape[1]))
            height, width = y2 - y1, x2 - x1

            if width > 0 and height > 0:
                batch_labels[label] = 1

                radius = gaussian_radius((math.ceil(width), math.ceil(height)))
                radius = max(0, int(radius))

                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)

                batch_hm[:, :, label] = draw_gaussian(batch_hm[:, :, label], center_int, radius)
                batch_wh[center_int[1], center_int[0]] = width, height
                batch_offset[center_int[1], center_int[0]] = center - center_int
                batch_offset_mask[center_int[1], center_int[0]] = 1

        item = [batch_labels, image, batch_hm, batch_wh, batch_offset, batch_offset_mask]
        item = list(map(lambda array: torch.Tensor(array), item))

        return item

    @property
    def read_data(self):
        # Reading
        print(f"Images: {self.image_folder}")
        image_files = os.listdir(self.image_folder)
        print(f"Annotations: {self.annotation_folder}")
        print("Reading data... ", end='')

        data = []
        total = 0
        for image_file in image_files:
            # Add image and preprocess
            if '.jpg' not in image_file:
                continue
            image = cv2.imread(os.path.join(self.image_folder, image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            datum = [image]

            # Add annotations
            name, ext = os.path.splitext(image_file)
            annotation_file = os.path.join(self.annotation_folder, f"{name}.txt")
            if not os.path.exists(annotation_file):
                # raise FileNotFoundError(f"{annotation_file} does not exist. ")
                print(f"{annotation_file} does not exist. ")
                destination = f'{self.image_folder}/../no_anno/'
                if not os.path.exists(destination):
                    os.mkdir(destination)
                shutil.move(os.path.join(self.image_folder, image_file), destination)
                continue
            with open(annotation_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                bboxes = []
                for line in lines:
                    # if eval(line.split()[0]) == 0:
                    #     bboxes.append(list(map(eval, line.split())))
                    #     self.classes[eval(line.split()[0])] += 1
                    bboxes.append(list(map(eval, line.split())))
                    self.classes[eval(line.split()[0])] += 1
                bboxes = np.array(bboxes)
            datum.append(bboxes)

            # Save datum to data
            data.append(datum)  # Each would be: [image, [[l, x, y, w, h], annotation2, ...]]
            total += 1

        print(f"Complete! Total: {total}")
        return data


if __name__ == "__main__":
    project_name = 'coco'
    data_set = CenterNetDataset(
        image_folder=f"./dataset/{project_name}/train/image/",
        annotation_folder=f"./dataset/{project_name}/train/annotation/",
        train=True,
        num_classes=80,
    )
    # test_data = train_set[0][1].cpu().detach().numpy()
    # test_data = test_data.transpose(1, 2, 0)
    # cv2.imshow('test', test_data)
    # cv2.waitKey(0)
    for data in data_set:
        test = data[1].cpu().detach().numpy().transpose(1, 2, 0)
        test = unnormalize_image(test).astype('uint8')
        test = cv2.cvtColor(test, cv2.COLOR_RGB2BGR)
        hm = cv2.resize(data[2][:, :, 0].cpu().detach().numpy(), (512, 512))
        # hm = data[2][:, :, 0]
        mask = cv2.resize(data[-1].cpu().detach().numpy(), (512, 512))
        # for row in hm:
        #     for element in row:
        #         print(element, end=' ')
        #     print()
        # print(hm.eq(1).float().sum())
        cv2.imshow('test', test)
        cv2.imshow('hm', hm)
        cv2.imshow('mask', mask)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
