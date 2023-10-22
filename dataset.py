import torch
from torch.utils.data.dataset import Dataset
import os
import cv2
import numpy as np
import math
from utils import gaussian_radius, draw_gaussian, normalize_image

class CenterNetDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, input_shape=(512, 512), stride=4, num_classes=1):
        super(CenterNetDataset, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (input_shape[0] // stride, input_shape[1] // stride)
        self.num_classes = num_classes

        self.red_list, self.blue_list, self.green_list = [], [], []
        self.data = []

        print(f"Reading images in {image_folder}... ")
        image_files = os.listdir(image_folder)
        print(f"Reading annotations in {annotation_folder}... ")
        annotation_files = os.listdir(annotation_folder)

        total = 0
        for image_file in image_files:
            # Add image and preprocess
            image = cv2.imread(image_folder + image_file)
            image = cv2.resize(image, self.input_shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            datum = [image]

            # Save colors
            r, g, b = image
            self.red_list.append(np.mean(r))
            self.green_list.append(np.mean(g))
            self.blue_list.append(np.mean(b))

            # Add annotations
            name = image_file[:image_file.find('.')]
            for annotation_file in annotation_files:
                if name in annotation_file:
                    with open(annotation_folder + annotation_file, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        annotations = []
                        for line in lines:
                            annotations.append(list(map(eval, line.split())))  # [label, x, y, w, h]
                    break

            datum.append(annotations)
            self.data.append(datum)  # Each would be: [image, [annotation1, annotation2, ...]]
            total += 1

        print(f"Complete! Total: {total}\n")
        self.mean, self.std = self.get_mean_std()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # Initial hm, wh, offset, mask
        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_wh = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_offset = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_offset_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)

        # Get image, annotation
        image, annotations = self.data[item]

        # Normalize image
        image = normalize_image(image, self.mean, self.std)
        image = torch.from_numpy(image).float()

        # Get true hm, wh, offset, mask
        for annotation in annotations:
            label, x, y, w, h = annotation
            x1 = (x - w / 2) * self.output_shape[0]
            x2 = (x + w / 2) * self.output_shape[0]
            y1 = (y - h / 2) * self.output_shape[1]
            y2 = (y + h / 2) * self.output_shape[1]
            width = w * self.output_shape[0]
            height = h * self.output_shape[1]
            # print(x1, y1, x2, y2)
            # test_image = cv2.resize(image, self.output_shape)
            # cv2.rectangle(test_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
            # cv2.imshow("1", test_image)

            if width > 0 and height > 0:
                radius = gaussian_radius((math.ceil(width), math.ceil(height)))
                radius = max(0, int(radius))

                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)

                batch_hm[:, :, label] = draw_gaussian(batch_hm[:, :, label], center_int, radius)
                # cv2.imshow("2", batch_hm[:, :, label] * 255)
                # print(np.sum(batch_hm[:, :, label]))
                batch_wh[center_int[1], center_int[0]] = width, height
                batch_offset[center_int[1], center_int[0]] = center - center_int
                batch_offset_mask[center_int[1], center_int[0]] = 1

        return image, batch_hm, batch_wh, batch_offset, batch_offset_mask

    def get_mean_std(self):
        mean = [
            np.mean(np.array(self.red_list)),
            np.mean(np.array(self.green_list)),
            np.mean(np.array(self.blue_list))
        ]
        std = [
            np.std(np.array(self.red_list)),
            np.std(np.array(self.green_list)),
            np.std(np.array(self.blue_list))
        ]
        return mean, std

if __name__ == "__main__":
    train_set = CenterNetDataset(
        image_folder="./dataset/cat/train/image/",
        annotation_folder="./dataset/cat/train/annotation/",
    )
    test_data = train_set[0][0].detach().numpy()
    # print(test_data)
    test_data = test_data.transpose(1, 2, 0)
    cv2.imshow('test', test_data)
    cv2.waitKey(0)

    # for data in train_set:
    #     test = data
    #     key = cv2.waitKey(0)
    #     if key == ord('q'):
    #         break
