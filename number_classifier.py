from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import json
import random
from jersey_number_dataset import JerseyNumberDataset, JerseyNumberMultitaskDataset, TrackletMultitaskDataset
from networks import JerseyNumberClassifier, SimpleJerseyNumberClassifier, JerseyNumberMulticlassClassifier

import time
import copy
import argparse
import os
from tqdm import tqdm

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #print(f"input and label sizes:{len(inputs), len(labels)}")
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(f"output size is {len(outputs)}")
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

ALPHA = 0.5
BETA = 0.15
GAMMA = 0.15
DELTA = 0.2
def train_multitask_model(model, optimizer, scheduler, train_dataset, val_dataset, num_epochs=25, batch_size=32):
    num_workers = 0 if os.name == 'nt' else 4

    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}

    criterion = nn.CrossEntropyLoss()

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, digits1, digits2, digit_counts in tqdm(dataloaders[phase], desc=f'{phase}'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                digits1 = digits1.to(device)
                digits2 = digits2.to(device)
                digit_counts = digit_counts.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    out1, out2, out3, out4 = model(inputs)
                    _, preds = torch.max(out1, 1)
                    loss = (ALPHA * criterion(out1, labels)
                          + BETA * criterion(out2, digits1)
                          + GAMMA * criterion(out3, digits2)
                          + DELTA * criterion(out4, digit_counts))

                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def test_model(model, subset, model_type = None):
    model.eval()
    running_corrects = 0
    # Iterate over data.
    temp_max = 500
    temp_count = 0
    for batch in dataloaders[subset]:
        if model_type == 'resnet34_multi':
            inputs, labels = batch[0], batch[1]
        else:
            inputs, labels = batch
        temp_count += len(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        torch.set_grad_enabled(False)
        if model_type == 'resnet34_multi':
            outputs, _, _, _ = model(inputs)
        else:
            outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        print(preds, labels.data)
        running_corrects += torch.sum(preds == labels.data)
        if subset == 'train' and temp_count >= temp_max:
            break

    print(temp_count, dataset_sizes[subset], running_corrects )
    epoch_acc = running_corrects.double() / temp_count

    print(f"Accuracy {subset}:{epoch_acc}")
    return epoch_acc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def split_tracklet_ids(gt_json_path, val_fraction=0.15, seed=42):
    """Split tracklet IDs into train/val sets (by tracklet, not by image, to avoid leakage)."""
    with open(gt_json_path, 'r') as f:
        gt: dict[str, int] = json.load(f)
    # Only include tracklets with valid labels (1-99)
    valid_ids: list[str] = [tid for tid, label in gt.items() if 0 < label < 100]
    random.seed(seed)
    random.shuffle(valid_ids)
    val_count = max(1, int(len(valid_ids) * val_fraction))
    val_ids = set(valid_ids[:val_count])  # type: ignore[no-matching-overload]
    train_ids = set(valid_ids[val_count:])  # type: ignore[no-matching-overload]
    print(f"Tracklet split: {len(train_ids)} train, {len(val_ids)} val")
    return train_ids, val_ids


# Non-STR method for number recognition - used for comparison
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', help='resnet34 or resnet34_multi')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--data', help='SoccerNet root dir (e.g. data/SoccerNet/jersey-2023)')
    parser.add_argument('--weights', help='path to save/load model weights')
    parser.add_argument('--original_weights', help='path to pretrained weights for fine-tuning')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='fraction of train tracklets to hold out for validation (default: 0.15)')
    parser.add_argument('--epochs', type=int, default=15, help='number of training epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--max_per_tracklet', type=int, default=50,
                        help='max images sampled per tracklet, 0=all (default: 50)')

    args = parser.parse_args()

    num_workers = 0 if os.name == 'nt' else 4

    if args.model_type == 'resnet34_multi':
        # Tracklet-based data loading for multi-task model
        train_gt = os.path.join(args.data, 'train', 'train', 'train_gt.json')
        train_images_dir = os.path.join(args.data, 'train', 'train', 'images')
        test_gt = os.path.join(args.data, 'test', 'test', 'test_gt.json')
        test_images_dir = os.path.join(args.data, 'test', 'test', 'images')

        train_ids, val_ids = split_tracklet_ids(train_gt, val_fraction=args.val_split)

        mpt = args.max_per_tracklet
        train_dataset = TrackletMultitaskDataset(train_gt, train_images_dir, mode='train', tracklet_ids=train_ids, max_per_tracklet=mpt)
        val_dataset = TrackletMultitaskDataset(train_gt, train_images_dir, mode='val', tracklet_ids=val_ids, max_per_tracklet=mpt)
        test_dataset = TrackletMultitaskDataset(test_gt, test_images_dir, mode='test', max_per_tracklet=mpt)

        # Build dataloaders/sizes for test_model()
        dataloaders = {
            'test': torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=num_workers)
        }
        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
    else:
        # Original flat CSV-based data loading
        train_img_dir = os.path.join(args.data, 'train', 'imgs')
        test_img_dir = os.path.join(args.data, 'test', 'imgs')
        val_img_dir = os.path.join(args.data, 'val', 'imgs')

        annotations_file_train = os.path.join(train_img_dir, 'train_gt.txt')
        annotations_file_val = os.path.join(val_img_dir, 'val_gt.txt')
        annotations_file_test = os.path.join(test_img_dir, 'test_gt.txt')

        image_dataset_train = JerseyNumberDataset(annotations_file_train, train_img_dir, 'train')
        image_dataset_test = JerseyNumberDataset(annotations_file_test, test_img_dir, 'test')
        image_dataset_val = JerseyNumberDataset(annotations_file_val, val_img_dir, 'val')

        dataloader_train = torch.utils.data.DataLoader(image_dataset_train, batch_size=4,
                                                       shuffle=True, num_workers=num_workers)
        dataloader_val = torch.utils.data.DataLoader(image_dataset_val, batch_size=4,
                                                     shuffle=True, num_workers=num_workers)
        dataloader_test = torch.utils.data.DataLoader(image_dataset_test, batch_size=4,
                                                      shuffle=False, num_workers=num_workers)

        image_datasets = {'train': image_dataset_train, 'val': image_dataset_val, 'test': image_dataset_test}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        dataloaders = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}

    if args.simple:
        model_ft = SimpleJerseyNumberClassifier()
    elif args.model_type == 'resnet34':
        model_ft = JerseyNumberClassifier()
    else:
        model_ft = JerseyNumberMulticlassClassifier()

    if args.fine_tune:
        state_dict = torch.load(args.original_weights, map_location=device)

    if args.train:
        model_ft = model_ft.to(device)
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        if args.model_type == 'resnet34_multi':
            model_ft = train_multitask_model(model_ft, optimizer_ft, exp_lr_scheduler,
                                             train_dataset, val_dataset,
                                             num_epochs=args.epochs, batch_size=args.batch_size)
        else:
            criterion = nn.CrossEntropyLoss()
            model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                   num_epochs=args.epochs)

        torch.save(model_ft.state_dict(), args.weights)

    else:  # test
        state_dict = torch.load(args.weights, map_location=device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        model_ft.load_state_dict(state_dict)
        model_ft = model_ft.to(device)
        test_model(model_ft, 'test', model_type=args.model_type)
