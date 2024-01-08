import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms
import cv2
import os
import shutil
from network import CenterNet
from utils import resize_image, normalize_image

def pool_nms(hm, kernel=3):
    pad = (kernel - 1) // 2
    hm_max = nn.functional.max_pool2d(hm, (kernel, kernel), stride=1, padding=pad)
    keep = (hm_max == hm).float()
    return hm * keep

def detect(origin_image):
    h, w = origin_image.shape[:2]
    global input_shape
    image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    image = resize_image(image, input_shape)
    image = np.transpose(normalize_image(image), (2, 0, 1))
    image_datum = np.expand_dims(image, 0)

    # Get prediction
    image = torch.from_numpy(image_datum).type(torch.FloatTensor)
    image = image.to(device)
    pred = model(image)

    # Decode bbox
    hm_pred, wh_pred, offset_pred = pred
    _, ih, iw, _ = hm_pred.shape

    hm_pred = pool_nms(hm_pred)[0].view(-1, num_classes)
    wh_pred = wh_pred[0].view(-1, 2)
    offset_pred = offset_pred[0].view(-1, 2)

    yv, xv = torch.meshgrid(torch.arange(0, ih), torch.arange(0, iw))
    xv, yv = map(lambda x: x.flatten().float().to(device), [xv, yv])

    class_conf, class_pred = torch.max(hm_pred, dim=-1)
    mask = class_conf > confidence
    wh_pred_mask = wh_pred[mask]
    offset_pred_mask = offset_pred[mask]

    xv_mask = torch.unsqueeze(xv[mask] + offset_pred_mask[..., 0], -1)
    yv_mask = torch.unsqueeze(yv[mask] + offset_pred_mask[..., 1], -1)
    half_w, half_h = wh_pred_mask[..., 0:1] / 2, wh_pred_mask[..., 1:2] / 2
    bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
    bboxes[:, [0, 2]] /= iw
    bboxes[:, [1, 3]] /= ih
    detection = torch.cat(
        [bboxes, torch.unsqueeze(class_conf[mask], -1), torch.unsqueeze(class_pred[mask], -1).float()], dim=-1)

    # Postprocess
    output = None
    unique_labels = detection[:, -1].cpu().unique()
    unique_labels, detection = map(lambda x: x.to(device), [unique_labels, detection])

    for c in unique_labels:
        detect_class = detection[detection[:, -1] == c]
        keep = nms(
            detect_class[:, :4],
            detect_class[:, 4],
            nms_thres
        )
        max_detect = detect_class[keep]
        output = max_detect if output is None else torch.cat((output, max_detect))

    if output is not None:
        output = output.cpu().numpy()
        box_xy, box_wh = (output[:, 0:2] + output[:, 2:4]) / 2, output[:, 2:4] - output[:, 0:2]

        # Correct boxes
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array((h, w))

        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)

        output[:, :4] = boxes

    # Process image
    result = output
    if result is None:
        return []
    top_label = np.array(result[:, 5], dtype='int32')
    top_conf = result[:, 4]
    top_boxes = result[:, :4]

    ps = []
    for i, c in list(enumerate(top_label)):
        box = top_boxes[i]
        score = top_conf[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(h, np.floor(bottom).astype('int32'))
        right = min(w, np.floor(right).astype('int32'))

        x = (left + right) / 2 / w
        y = (top + bottom) / 2 / h
        w_ = (right - left) / w
        h_ = (bottom - top) / h
        ps.append([int(c), score, x, y, w_, h_])

    return ps

def calculate_iou(box1, box2):
    # 计算两个框的IoU
    x1, y1, w1, h1 = box1[2], box1[3], box1[4], box1[5]
    x2, y2, w2, h2 = box2[2], box2[3], box2[4], box2[5]

    intersection_x = max(0, min(x1 + w1/2, x2 + w2/2) - max(x1 - w1/2, x2 - w2/2))
    intersection_y = max(0, min(y1 + h1/2, y2 + h2/2) - max(y1 - h1/2, y2 - h2/2))

    intersection = intersection_x * intersection_y
    union = w1 * h1 + w2 * h2 - intersection

    iou = intersection / (union + 1e-16)
    return iou

def calculate_ap(precision, recall):
    # 计算平均精度
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            ap += np.max(precision[mask])
    ap /= 11
    return ap

def calculate_mAP(predictions, ground_truths, iou_threshold=0.50):
    # 计算mAP
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # 根据置信度排序预测
    predictions = predictions[predictions[:, 1].argsort()[::-1]]

    true_positives = np.zeros(len(predictions))
    false_positives = np.zeros(len(predictions))

    for i, prediction in enumerate(predictions):
        ious = [calculate_iou(prediction, gt) for gt in ground_truths]
        max_iou = np.max(ious)
        max_iou_index = np.argmax(ious)

        if max_iou >= iou_threshold and ground_truths[max_iou_index][0] == prediction[0]:
            if not ground_truths[max_iou_index][-1]:  # Check if the true positive is not already matched
                true_positives[i] = 1
                ground_truths[max_iou_index][-1] = 1  # Mark the true positive as already matched
            else:
                false_positives[i] = 1
        else:
            false_positives[i] = 1

    cumulative_true_positives = np.cumsum(true_positives)
    cumulative_false_positives = np.cumsum(false_positives)

    precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives + 1e-16)
    recall = cumulative_true_positives / len(ground_truths)

    ap = calculate_ap(precision, recall)
    return ap

if __name__ == "__main__":
    project_name = 'coco'
    weight_path = './weights/save/0108.pth'
    image_folder = f'./dataset/coco/val/image/'
    annotation_folder = "./dataset/coco/val/annotation"
    confidence = 0.1
    nms_thres = 0.4

    input_shape = (512, 512)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    with open(f"./dataset/{project_name}/name.txt", 'r') as file:
        classes = list(map(lambda x: x.strip(), file.readlines()))
    print("Class names: ", end='')
    num_classes = 0
    for cls in classes:
        if num_classes % 10 == 0:
            print('\n\t', end='')
        print(f" [{cls}] ", end='')
        num_classes += 1
    print(f"\nTotal: {num_classes}\n")

    # Load model
    model = CenterNet(num_classes=num_classes)
    model.to(device)
    model.eval()
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight_path).items()})

    mAP = 0
    with torch.no_grad():
        image_files = os.listdir(image_folder)
        for count, image_file in enumerate(image_files):
            predictions = []
            ground_truths = []

            # Load images
            image_data = cv2.imread(os.path.join(image_folder, image_file))
            if '.jpg' not in image_file:
                continue
            for prediction in detect(image_data):
                predictions.append(prediction)

            # Load ground truth
            name, ext = os.path.splitext(image_file)
            annotation_file = os.path.join(annotation_folder, f"{name}.txt")
            if not os.path.exists(annotation_file):
                # raise FileNotFoundError(f"{annotation_file} does not exist. ")
                print(f"{annotation_file} does not exist. ")
                destination = f'{image_folder}/../no_anno/'
                if not os.path.exists(destination):
                    os.mkdir(destination)
                shutil.move(os.path.join(image_folder, image_file), destination)
                continue
            with open(annotation_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    gt = list(map(eval, line.split()))
                    gt.append(0)
                    gt.insert(1, 1)
                    ground_truths.append(gt)

            mAP += calculate_mAP(predictions, ground_truths)
        mAP /= len(image_files)
        print(f"mAP: {mAP}")
