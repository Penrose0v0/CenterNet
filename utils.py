import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def draw_figure(x, y, title, save_path):
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()

def image_resize(image, target_size, gt_boxes=None):
    iw, ih = target_size

    h, w = image.shape[:2]

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=np.uint8)
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized

    if gt_boxes is None:
        return image_paded
    elif gt_boxes.size == 0:
        # Use no label image to train
        return image_paded, gt_boxes
    else:
        new_boxes = gt_boxes.copy()
        new_boxes[:, 1] = (gt_boxes[:, 1] * nw + dw) / iw
        new_boxes[:, 2] = (gt_boxes[:, 2] * nh + dh) / ih
        new_boxes[:, 3] = gt_boxes[:, 3] * nw / iw
        new_boxes[:, 4] = gt_boxes[:, 4] * nh / ih
        return image_paded, new_boxes

def normalize_image(image, mean, std):
    image = image.astype(np.float32)
    for channel in range(3):
        image[channel] = (image[channel] - mean[channel]) / std[channel]
        image[channel] = np.clip(image[channel], -1.0, 1.0)
    return image


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)



if __name__ == "__main__":
    image_folder = "dataset/cat/val/image/"
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        img = cv2.imread(os.path.join(image_folder, image_file))
        print(img.shape)
        input_size = (512, 512)

        cv2.imshow('origin', img)
        m = get_affine_matrix(img, input_size)
        print(m[0][0])
        print(img.shape[0] * m[0][0] / 4)

        img = cv2.warpAffine(img, m, input_size)
        img = cv2.resize(img, (128, 128))
        print(img.shape)
        cv2.imshow('transformed', img)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break
