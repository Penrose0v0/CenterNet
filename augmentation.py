import numpy as np
import random
import cv2

def data_augmentation(origin_image, origin_bboxes):
    image, bboxes = origin_image.copy(), origin_bboxes.copy()
    if random.random() < 0.5:
        image, bboxes = random_horizontal_flip(image, bboxes)
    if random.random() < 0.5:
        image, bboxes = random_vertical_flip(image, bboxes)
    if random.random() < 0.5:
        image, bboxes = random_crop(image, bboxes)
    if random.random() < 0.5:
        image, bboxes = random_translate(image, bboxes)

    # Brightness
    if random.random() < 0.5:
        brightness_factor = random.uniform(0.6, 1.4)
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        image = np.clip(image, 0, 255)

    # Color
    if random.random() < 0.5:
        color_factor = random.uniform(0.7, 1.3)
        zeros = np.zeros(image.shape, dtype=np.uint8)
        image = cv2.addWeighted(image, color_factor, zeros, 1 - color_factor, 0)
        image = np.clip(image, 0, 255)

    # Contrast
    if random.random() < 0.5:
        contrast_factor = random.uniform(0.7, 1.3)
        image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
        image = np.clip(image, 0, 255)

    # Sharpness
    if random.random() < 0.5:
        sharpness_factor = random.uniform(0, 2.0)
        kernel = np.array([[-1, -1, -1],
                           [-1, 9 + sharpness_factor, -1],
                           [-1, -1, -1]])
        image = cv2.filter2D(image, -1, kernel)
        image = np.clip(image, 0, 255)

    return image, bboxes


def random_horizontal_flip(image, bboxes):
    image = image[:, ::-1, :]
    image = np.array(image)

    if bboxes.size != 0:
        bboxes[:, 1] = 1 - bboxes[:, 1]

    return image, bboxes


def random_vertical_flip(image, bboxes):
    image = image[::-1, :, :]
    image = np.array(image)

    if bboxes.size != 0:
        bboxes[:, 2] = 1 - bboxes[:, 2]

    return image, bboxes

def random_crop(image, bboxes):
    if bboxes.size == 0:
        return image, bboxes

    h, w, _ = image.shape

    # 创建一个新的边界框数组，复制原始数据
    new_bboxes = np.copy(bboxes)

    # 转换新的边界框格式到旧的格式
    new_bboxes[:, 0] = (bboxes[:, 1] - 0.5 * new_bboxes[:, 3]) * w
    new_bboxes[:, 1] = (bboxes[:, 2] - 0.5 * new_bboxes[:, 4]) * h
    new_bboxes[:, 2] = (bboxes[:, 1] + 0.5 * new_bboxes[:, 3]) * w
    new_bboxes[:, 3] = (bboxes[:, 2] + 0.5 * new_bboxes[:, 4]) * h

    max_bbox = np.concatenate([np.min(new_bboxes[:, 0:2], axis=0), np.max(new_bboxes[:, 2:4], axis=0)], axis=-1)

    max_l_trans = max_bbox[0]
    max_u_trans = max_bbox[1]
    max_r_trans = w - max_bbox[2]
    max_d_trans = h - max_bbox[3]

    crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
    crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
    crop_xmax = min(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
    crop_ymax = min(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

    image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
    h, w, _ = image.shape

    # 更新新的边界框坐标
    new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]] - crop_xmin
    new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]] - crop_ymin

    # 转换边界框格式回新的格式
    bboxes[:, 1] = (new_bboxes[:, 0] + new_bboxes[:, 2]) / (2 * w)
    bboxes[:, 2] = (new_bboxes[:, 1] + new_bboxes[:, 3]) / (2 * h)
    bboxes[:, 3] = (new_bboxes[:, 2] - new_bboxes[:, 0]) / w
    bboxes[:, 4] = (new_bboxes[:, 3] - new_bboxes[:, 1]) / h

    return image, bboxes

def random_translate(image, bboxes):
    if bboxes.size == 0:
        return image, bboxes

    h, w, _ = image.shape

    # 创建一个新的边界框数组，复制原始数据
    new_bboxes = np.copy(bboxes)

    # 转换新的边界框格式到旧的格式
    new_bboxes[:, 0] = (bboxes[:, 1] - 0.5 * new_bboxes[:, 3]) * w
    new_bboxes[:, 1] = (bboxes[:, 2] - 0.5 * new_bboxes[:, 4]) * h
    new_bboxes[:, 2] = (bboxes[:, 1] + 0.5 * new_bboxes[:, 3]) * w
    new_bboxes[:, 3] = (bboxes[:, 2] + 0.5 * new_bboxes[:, 4]) * h

    max_bbox = np.concatenate([np.min(new_bboxes[:, 0:2], axis=0), np.max(new_bboxes[:, 2:4], axis=0)], axis=-1)
    max_l_trans = max_bbox[0]
    max_u_trans = max_bbox[1]
    max_r_trans = w - max_bbox[2]
    max_d_trans = h - max_bbox[3]

    tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
    ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

    M = np.array([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (w, h), borderValue=(128, 128, 128))

    # 更新新的边界框坐标
    new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]] + tx
    new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]] + ty

    # 转换边界框格式回新的格式
    bboxes[:, 1] = (new_bboxes[:, 0] + new_bboxes[:, 2]) / (2 * w)
    bboxes[:, 2] = (new_bboxes[:, 1] + new_bboxes[:, 3]) / (2 * h)
    bboxes[:, 3] = (new_bboxes[:, 2] - new_bboxes[:, 0]) / w
    bboxes[:, 4] = (new_bboxes[:, 3] - new_bboxes[:, 1]) / h

    return image, bboxes

if __name__ == "__main__":
    import os
    image_folder = "./dataset/coco/train/image/"
    image_files = os.listdir(image_folder)
    annotation_folder = "./dataset/coco/train/annotation/"
    for image_file in image_files:
        image = cv2.imread(os.path.join(image_folder, image_file))

        name, ext = os.path.splitext(image_file)
        annotation_file = os.path.join(annotation_folder, f"{name}.txt")
        with open(annotation_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            bboxes = []
            for line in lines:
                bboxes.append(list(map(eval, line.split())))
        bboxes = np.array(bboxes)
        backup = image.copy()

        height, width, _ = image.shape
        for bbox in bboxes:
            _, x, y, w, h = bbox
            cv2.rectangle(image,
                          (int((x - w / 2) * width), int((y - h / 2) * height)),
                          (int((x + w / 2) * width), int((y + h / 2) * height)),
                          (0, 0, 255))
        cv2.imshow("origin", image)

        new_image, new_bboxes = data_augmentation(backup, bboxes)

        height, width, _ = new_image.shape
        for bbox in new_bboxes:
            _, x, y, w, h = bbox
            cv2.rectangle(new_image,
                          (int((x - w / 2) * width), int((y - h / 2) * height)),
                          (int((x + w / 2) * width), int((y + h / 2) * height)),
                          (255, 0, 0))
        cv2.imshow("trans", new_image)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        print()
