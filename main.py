import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import os
import argparse

from network import CenterNet
from loss import CenterNetLoss
from dataset import CenterNetDataset
from utils import draw_figure

def train(count=1):
    running_loss, running_loss_k, running_loss_off, running_loss_size = 0.0, 0.0, 0.0, 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # Get data
        image, hm_target, wh_target, offset_target, offset_mask = map(lambda x: x.to(device), data)
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
            print(f"Batch {batch_idx + 1:<5d} "
                  f"Loss = {running_loss / count:<10.4f} "
                  f"Loss_k = {running_loss_k / count:<10.4f} "
                  f"Loss_off = {running_loss_off / count:<10.4f} "
                  f"Loss_size = {running_loss_size / count:<10.4f}")
            running_loss, running_loss_k, running_loss_off, running_loss_size = 0.0, 0.0, 0.0, 0.0

def val(epoch_num):
    running_loss, running_loss_k, running_loss_off, running_loss_size = 0.0, 0.0, 0.0, 0.0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            # Get data
            image, hm_target, wh_target, offset_target, offset_mask = map(lambda x: x.to(device), data)
            target = hm_target, wh_target, offset_target, offset_mask

            # Forward only
            pred = model(image)
            loss, loss_k, loss_off, loss_size = criterion(pred, target)

            # Calculate loss
            running_loss += loss.item()
            running_loss_k += loss_k.item()
            running_loss_off += loss_off.item()
            running_loss_size += loss_size.item()
            total += 1

            # Save result
            if epoch_num % 10 != 9:
                continue
            save_path = f'./outputs/epoch {epoch_num + 1}'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
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

                # Save images
                cv2.imwrite(save_path + f"/{count} origin.jpg", origin_save)
                cv2.imwrite(save_path + f"/{count} hm_pred.jpg", hm_pred_save)
                cv2.imwrite(save_path + f"/{count} hm_true.jpg", hm_true_save)
                count += 1


    val_loss = running_loss / total
    val_loss_k = running_loss_k / total
    val_loss_off = running_loss_off / total
    val_loss_size = running_loss_size / total
    print(f"Validation Loss = {val_loss:.4f}")
    return val_loss, val_loss_k, val_loss_off, val_loss_size

if __name__ == "__main__":
    fmt = "----- {:^25} -----"
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--epochs', type=int, default=140)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--min-confidence', type=float, default=0.25)
    parser.add_argument('--project-name', type=str, default='coco')
    args = parser.parse_args()

    # Set hyper-parameters
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    min_confidence = args.min_confidence
    project_name = args.project_name

    # Set device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Analyze task
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
    print(f"\nTotal: {num_classes}\n")

    # Create neural network
    model = CenterNet(num_classes=num_classes)
    model.to(device)
    weights = args.weights
    if weights != '':
        model.load_state_dict(torch.load(weights))
        print(fmt.format("Load pretrained model") + '\n')

    # Load dataset
    print(fmt.format("Loading training set"))
    train_set = CenterNetDataset(
        image_folder=f"./dataset/{project_name}/train/image/",
        annotation_folder=f"./dataset/{project_name}/train/annotation/",
        num_classes=num_classes,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    print(fmt.format("Loading validation set"))
    val_set = CenterNetDataset(
        image_folder=f"./dataset/{project_name}/val/image/",
        annotation_folder=f"./dataset/{project_name}/val/annotation/",
        num_classes=num_classes,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # Define criterion and optimizer
    criterion = CenterNetLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    print(fmt.format("Start training") + '\n')
    min_loss = 100
    epoch_list, loss_list, loss_k_list, loss_off_list, loss_size_list = [], [], [], [], []
    for epoch in range(epochs):
        print(f"< Epoch {epoch + 1} >")

        # Train + Val
        train()
        current_loss, current_loss_k, current_loss_off, current_loss_size = val(epoch)

        # Save model
        torch.save(model.state_dict(), f"./weights/epoch_{epoch + 1}.pth")
        if current_loss < min_loss:
            torch.save(model.state_dict(), f"./weights/best.pth")
            print("Update the best model")
            min_loss = current_loss
        print()

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

    print(f"Training finished! Min Loss: {min_loss:.4f}")
