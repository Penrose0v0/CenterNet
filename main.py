import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import os

from network import CenterNet
from loss import CenterNetLoss
from dataset import CenterNetDataset

# Hyper parameter
batch_size = 16
learning_rate = 0.001
epochs = 140
num_classes = 1
min_confidence = 0.25

# Set device
device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# Create neural network
model = CenterNet(num_classes=num_classes)
model.to(device)


# Load dataset
train_set = CenterNetDataset(
    image_folder="./dataset/cat/train/image/",
    annotation_folder="./dataset/cat/train/annotation/",
    num_classes=num_classes,
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

val_set = CenterNetDataset(
    image_folder="./dataset/cat/val/image/",
    annotation_folder="./dataset/cat/val/annotation/",
    num_classes=num_classes,
)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)


# Define criterion and optimizer
criterion = CenterNetLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch_num, count=1):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        image, hm_target, wh_target, offset_target, offset_mask = map(lambda x: x.to(device), data)
        target = hm_target, wh_target, offset_target, offset_mask
        optimizer.zero_grad()

        # Forward + Backward + Update
        pred = model(image)
        loss, loss_k, loss_off, loss_size = criterion(pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % count == count - 1:
            print(f"Epoch {epoch_num + 1: <4d} Batch {batch_idx + 1: <4d} Loss = {running_loss / count:.4f}\t"
                  f"Loss_k = {loss_k:.4f} \tLoss_off = {loss_off:.4f} \tLoss_size = {loss_size:.4f}")
            running_loss = 0.0

def val(epoch_num):
    running_loss = 0
    total = 0
    if not os.path.exists(f'.\\outputs\\epoch {epoch_num + 1}'):
        os.mkdir(f'.\\outputs\\epoch {epoch_num + 1}')
    with torch.no_grad():
        for data in val_loader:
            # Validation
            image, hm_target, wh_target, offset_target, offset_mask = map(lambda x: x.to(device), data)
            target = hm_target, wh_target, offset_target, offset_mask
            pred = model(image)
            loss, loss_k, loss_off, loss_size = criterion(pred, target)

            running_loss += loss.item()
            total += 1

            # Save result
            count = 1
            for origin, hm_true, hm_pred in zip(image, hm_target, pred[0]):
                # Post-process image
                origin_save = origin.detach().numpy().transpose(1, 2, 0)
                for i in range(3):
                    origin_save[i] = origin_save[i] * val_set.std + val_set.mean
                origin_save = origin_save.astype('uint8')

                # Post-process hm_pred
                hm_pred_save = cv2.resize(hm_pred.detach().numpy(), (512, 512))
                hm_pred_save = (hm_pred_save * 255).astype('uint8')

                # Post-process hm_true
                hm_true_save = cv2.resize(hm_true.detach().numpy(), (512, 512))
                hm_true_save = (hm_true_save * 255).astype('uint8')

                cv2.imwrite(f"./outputs/epoch {epoch_num + 1}/{count} origin.jpg", origin_save)
                cv2.imwrite(f"./outputs/epoch {epoch_num + 1}/{count} hm_pred.jpg", hm_pred_save)
                cv2.imwrite(f"./outputs/epoch {epoch_num + 1}/{count} hm_true.jpg", hm_true_save)
                count += 1
            print("Result saved")

    val_loss = running_loss / total
    print(f"Validation Loss = {val_loss:.4f}")
    return val_loss


def predict(epoch_num):
    input_shape = (512, 512)
    stride = 4
    output_shape = (input_shape[0] / stride, input_shape[1] / stride)

    # Preprocess the image
    original_image = cv2.imread("./dataset/cat/image.jpg")
    height, width, _ = original_image.shape
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).float().unsqueeze(0)

    # Draw bounding boxes
    hm, wh, offset = model(image)
    for cls in range(hm.shape[-1]):
        for y in range(1, hm.shape[1]-1):
            for x in range(1, hm.shape[2]-1):
                if hm[0, y, x, cls] > hm[0, y-1:y+2, x-1:x+2, cls].max() and hm[0, y, x, cls] >= min_confidence:
                    h, w = wh[y, x]
                    dy, dx = offset[y, x]
                    bbox_x1 = (x + dx - w / 2) * width / output_shape[1]
                    bbox_x2 = (x + dx + w / 2) * width / output_shape[1]
                    bbox_y1 = (y + dy - h / 2) * height / output_shape[0]
                    bbox_y2 = (y + dy + h / 2) * height / output_shape[0]
                    cv2.rectangle(original_image, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 0, 255))
    # print(hm[0, :, :, 0].detach().numpy() * 255)
    # hotmap = hm[0, :, :, 0].detach().numpy() * 255
    # hotmap = hotmap.astype('uint8')
    hotmap = cv2.resize(hm[0, :, :, 0].detach().numpy(), (height, width))
    print(hotmap)
    print(hotmap.shape)

    cv2.imshow(f"{epoch_num}", original_image)
    cv2.imshow(f"{epoch_num} hm", hotmap)

for epoch in range(epochs):
    train(epoch)
    val(epoch)
    torch.save(model.state_dict(), f"./weights/epoch_{epoch + 1}.pth")
    print()
    # predict(epoch)
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     break
    # cv2.destroyAllWindows()

# predict(1)
# key = cv2.waitKey(0)
