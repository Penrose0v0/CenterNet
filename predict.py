import os
import numpy as np
import cv2
import torch
from network import CenterNet
from utils import resize_image, normalize_image, unnormalize_image

device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
project_name = 'cat'

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
weight_path = './weights/current.pth'
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight_path).items()})

# Load images
image_folder = f'./dataset/{project_name}/train/image/'
image_files = os.listdir(image_folder)
image_data = []
for image_file in image_files:
    image = cv2.imread(os.path.join(image_folder, image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image, (512, 512))
    image = np.transpose(normalize_image(image), (2, 0, 1))
    image_datum = np.expand_dims(image, 0)
    image_data.append(image_datum)
image_data = np.array(image_data)

# Predict
with torch.no_grad():
    for count, image_datum in enumerate(image_data):
        image = torch.from_numpy(image_datum).type(torch.FloatTensor)
        image = image.to(device)
        pred = model(image)
        hm_pred = pred[0]

        save_path = f'./predict'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        image_save = image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        image_save = unnormalize_image(image_save).astype('uint8')

        for i in range(num_classes):
            hm_pred_save = cv2.resize(hm_pred.squeeze(0)[:, :, i].cpu().detach().numpy(), (512, 512))
            hm_pred_save = (hm_pred_save * 255).astype('uint8')

            image_save = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path + f"/{count + 1} origin.jpg", image_save)

            hm_pred_save = cv2.cvtColor(hm_pred_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path + f"/{count + 1}-{i} hm_pred.jpg", hm_pred_save)
