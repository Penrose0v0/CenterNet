import os
import colorsys
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.ops import nms
from network import CenterNet
from utils import resize_image, normalize_image

def pool_nms(hm, kernel=3):
    pad = (kernel - 1) // 2
    hm_max = nn.functional.max_pool2d(hm, (kernel, kernel), stride=1, padding=pad)
    keep = (hm_max == hm).float()
    return hm * keep

# Detect
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
        return origin_image
    top_label = np.array(result[:, 5], dtype='int32')
    top_conf = result[:, 4]
    top_boxes = result[:, :4]

    thickness = max((np.shape(image)[0] + np.shape(image)[1]) // 512, 1)

    for i, c in list(enumerate(top_label)):
        predicted_class = classes[int(c)]
        box = top_boxes[i]
        score = top_conf[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(h, np.floor(bottom).astype('int32'))
        right = min(w, np.floor(right).astype('int32'))

        print(predicted_class, top, left, bottom, right, score)

        label = '{} {:.2f}'.format(predicted_class, score)
        label_size, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label = label.encode('utf-8')

        if top - label_size[1] >= 0:
            text_origin = (left, top)
            rect_origin = (left, top - label_size[1])
        else:
            text_origin = (left, top + label_size[1])
            rect_origin = (left, top + 1)

        for i in range(thickness):
            cv2.rectangle(origin_image, (left, top), (right, bottom), colors[c], 1)

        cv2.rectangle(origin_image, tuple(rect_origin), tuple((np.array(rect_origin) + label_size).astype(int)),
                      colors[c], -1)
        font_color = (0, 0, 0) if np.array(colors[c]).mean() > 150 else (255, 255, 255)
        cv2.putText(origin_image, str(label, 'UTF-8'), text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)
    print()

    return origin_image

if __name__ == "__main__":
    project_name = 'coco'
    weight_path = './weights/save/0102.pth'
    image_folder = f'./img'
    # image_folder = f'./img'
    confidence = 0.3
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

    # Set color
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    # Load model
    model = CenterNet(num_classes=num_classes)
    model.to(device)
    model.eval()
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight_path).items()})

    with torch.no_grad():
        # Detect images
        image_files = os.listdir(image_folder)
        for count, image_file in enumerate(image_files):
            # Load images
            image_data = cv2.imread(os.path.join(image_folder, image_file))
            # origin_image = cv2.resize(origin_image, (origin_image.shape[1] // 5, origin_image.shape[0] // 5))
            image_data = detect(image_data)
            cv2.imshow('detection', image_data)
            key = cv2.waitKey(0)

            if key == ord('q'):
                break

        # # Detect stream
        # camera = cv2.VideoCapture(0)
        # while True:
        #     success, frame = camera.read()
        #
        #     if not success:
        #         print("Failed. ")
        #         break
        #
        #     frame = detect(frame)
        #     cv2.imshow('detection', frame)
        #     key = cv2.waitKey(1)
        #
        #     if key == ord('q'):
        #         break
