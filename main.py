import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import os
import argparse
import time

from network import CenterNet
from loss import CenterNetLoss
from dataset import CenterNetDataset
from utils import draw_figure, unnormalize_image

def train(epoch_num, count=100):
    model.train()
    running_loss, running_loss_k, running_loss_off, running_loss_size = 0.0, 0.0, 0.0, 0.0
    print(f"< Epoch {epoch_num + 1} >")
    for batch_idx, data in enumerate(train_loader, 0):
        # Get data
        image, hm_target, wh_target, offset_target, offset_mask = map(lambda x: x.to(device), data[1:])
        target = hm_target, wh_target, offset_target, offset_mask
        optimizer.zero_grad()

        # Forward + Backward + Update
        pred = model(image)
        loss, loss_k, loss_off, loss_size = criterion(pred, target)
        loss.backward()
        optimizer.step()

        # Calculate loss
        running_loss += loss.item()
        running_loss_k += loss_k.item()
        running_loss_off += loss_off.item()
        running_loss_size += loss_size.item()
        if batch_idx % count == count - 1:
            print(f"Batch {batch_idx + 1:<3d}\t\t"
                  f"Loss = {running_loss / count:.4f}\t\t"
                  f"Loss_k = {running_loss_k / count:.4f}\t\t"
                  f"Loss_off = {running_loss_off / count:.4f}\t\t"
                  f"Loss_size = {running_loss_size / count:.4f}")
            running_loss, running_loss_k, running_loss_off, running_loss_size = 0.0, 0.0, 0.0, 0.0

def val(epoch_num):
    model.eval()
    running_loss, running_loss_k, running_loss_off, running_loss_size = 0.0, 0.0, 0.0, 0.0
    total = 0
    with torch.no_grad():
        count = 1
        for data in val_loader:
            # Get data
            labels_s, images, hms_target, whs_target, offsets_target, offset_masks = map(lambda x: x.to(device), data)
            target = hms_target, whs_target, offsets_target, offset_masks

            # Forward only
            pred = model(images)
            loss, loss_k, loss_off, loss_size = criterion(pred, target)

            # Calculate loss
            running_loss += loss.item()
            running_loss_k += loss_k.item()
            running_loss_off += loss_off.item()
            running_loss_size += loss_size.item()
            total += 1

            # Save result
            if epoch_num % 1 != 0:
                continue
            save_path = f'./outputs/epoch {epoch_num + 1}'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            for image, labels, hm_target, hm_pred in zip(images, labels_s, hms_target, pred[0]):
                # Post-process image
                image_save = image.cpu().detach().numpy().transpose(1, 2, 0)
                image_save = unnormalize_image(image_save).astype('uint8')

                # Save images
                image_save = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path + f"/{count} origin.jpg", image_save)

                for i in range(num_classes):
                    label = int(labels[i].cpu().detach().numpy())
                    if label != 1:
                        continue

                    # Post-process hm_pred
                    hm_pred_save = cv2.resize(hm_pred[:, :, i].cpu().detach().numpy(), (512, 512))
                    hm_pred_save = (hm_pred_save * 255).astype('uint8')

                    # Post-process hm_target
                    hm_true_save = cv2.resize(hm_target[:, :, i].cpu().detach().numpy(), (512, 512))
                    hm_true_save = (hm_true_save * 255).astype('uint8')

                    # Save heatmap
                    hm_pred_save = cv2.cvtColor(hm_pred_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path + f"/{count}-{i} hm_pred.jpg", hm_pred_save)

                    hm_true_save = cv2.cvtColor(hm_true_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path + f"/{count}-{i} hm_true.jpg", hm_true_save)
                count += 1

    val_loss = running_loss / total
    val_loss_k = running_loss_k / total
    val_loss_off = running_loss_off / total
    val_loss_size = running_loss_size / total
    print(f"Validation\t\t"
          f"Loss = {val_loss:.4f}\t\t"
          f"Loss_k = {val_loss_k:.4f}\t\t"
          f"Loss_off = {val_loss_off:.4f}\t\t"
          f"Loss_size = {val_loss_size:.4f}")
    return val_loss, val_loss_k, val_loss_off, val_loss_size

if __name__ == "__main__":
    fmt = "----- {:^25} -----"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--backbone-only', type=bool, default=False)
    parser.add_argument('--freeze-epochs', type=int, default=100)
    parser.add_argument('--unfreeze-epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--min-confidence', type=float, default=0.25)
    parser.add_argument('--project-name', type=str, default='coco')
    args = parser.parse_args()

    """Set hyper-parameters"""
    freeze_epochs = args.freeze_epochs
    unfreeze_epochs = args.unfreeze_epochs
    epochs = freeze_epochs + unfreeze_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    min_confidence = args.min_confidence
    project_name = args.project_name
    model_path = args.model_path
    backbone_only = args.backbone_only

    # Set device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    """Analyze task"""
    print(fmt.format("Analyzing task"))
    with open(f"./dataset/{project_name}/name.txt", 'r') as file:
        classes = list(map(lambda x: x.strip(), file.readlines()))
    print("Class names: ", end='')
    num_classes = 0
    for cls in classes:
        if num_classes % 10 == 0:
            print('\n\t', end='')
        print(f" [{cls}] ", end='')
        num_classes += 1
    # num_classes = 1
    print(f"\nTotal: {num_classes}\n")

    """Create neural network"""
    print(fmt.format("Create neural network"))

    # Create an instance
    model = CenterNet(num_classes=num_classes)
    device_count = torch.cuda.device_count()
    print(f"Using {device_count} GPUs")

    # Load pretrained model or create a new model
    if model_path != '':
        print(f"Loading pretrained model: {model_path}")
        print(f"Backbone only: {backbone_only}")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        loading_dict = {}
        unloaded_layers = []
        for k, v in pretrained_dict.items():
            k = k.replace('module.', '')
            if backbone_only and 'backbone.' not in k:
                continue
            if k not in model_dict:
                unloaded_layers.append(k)
                continue
            loading_dict[k] = v
        print(f"Loaded layers: {loading_dict.keys()}")
        print(f"Unloaded layers: {unloaded_layers}")
        model_dict.update(loading_dict)
        model.load_state_dict(model_dict)
    else:
        print("Creating new model")
    model = nn.DataParallel(model)
    model.to(device)
    print()

    # Define criterion
    criterion = CenterNetLoss()
    optimizer = optim.Adam(params=model.parameters())

    """Load dataset"""
    # Load train set
    print(fmt.format("Loading training set"))
    train_set = CenterNetDataset(
        image_folder=f"./dataset/{project_name}/train/image/",
        annotation_folder=f"./dataset/{project_name}/train/annotation/",
        train=True,
        num_classes=num_classes,
    )
    print("Number of sample for each class: ", end='')
    for i in range(num_classes):
        if i % 10 == 0:
            print('\n\t', end='')
        print(f' {classes[i]}: {train_set.classes[i]} ', end='')
    print('\n')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # Load val set
    print(fmt.format("Loading validation set"))
    val_set = CenterNetDataset(
        image_folder=f"./dataset/{project_name}/val/image/",
        annotation_folder=f"./dataset/{project_name}/val/annotation/",
        train=False,
        num_classes=num_classes,
    )
    print()
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)

    """Start training"""
    min_loss = -1
    best_epoch = 0
    epoch_list, loss_list, loss_k_list, loss_off_list, loss_size_list = [], [], [], [], []
    for epoch in range(epochs):
        start = time.time()

        # Reset optimizer
        if epoch == 0 and freeze_epochs > 0:
            print(fmt.format("Start freeze training") + '\n')
            model.module.freeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        if epoch == freeze_epochs:
            print(fmt.format("Start unfreeze training") + '\n')
            model.module.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

        # Train + Val
        train(epoch)
        current_loss, current_loss_k, current_loss_off, current_loss_size = val(epoch)

        # Save model
        torch.save(model.state_dict(), f"./weights/current.pth")
        if current_loss < min_loss or min_loss == -1:
            torch.save(model.state_dict(), f"./weights/best.pth")
            print("Update the best model")
            min_loss = current_loss
            best_epoch = epoch + 1

        # Draw figure
        epoch_list.append(epoch + 1)
        loss_list.append(current_loss)
        loss_k_list.append(current_loss_k)
        loss_off_list.append(current_loss_off)
        loss_size_list.append(current_loss_size)
        draw_figure(epoch_list, loss_list, "Loss", "./outputs/loss.png")
        draw_figure(epoch_list, loss_k_list, "Loss_k", "./outputs/loss_k.png")
        draw_figure(epoch_list, loss_off_list, "Loss_off", "./outputs/loss_off.png")
        draw_figure(epoch_list, loss_size_list, "Loss_size", "./outputs/loss_size.png")

        # Elapsed time
        end = time.time()
        use_time = int(end - start)
        print(f"Elapsed time: {use_time // 60}m {use_time % 60}s\n")

    print(f"Training finished! Best Epoch: {best_epoch}, Min Loss: {min_loss:.4f}")
