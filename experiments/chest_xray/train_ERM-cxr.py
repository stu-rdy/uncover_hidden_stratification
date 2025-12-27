from torch.utils.data import Dataset, Subset, DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, accuracy_score, recall_score, precision_score

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import wandb
import copy 
from src.data_loader import CSVDataset, CSVDatasetWithName

def save_checkpoint(model, epoch, name):
    model_out_path = "../../saves/" + name + ".pth"
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("../../saves/"):
        os.makedirs("../../saves/")

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))

def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)
    
def extract_features(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    #x = model.fc(x)
    return x

def train_epoch_ERM(model, optimizer, criterion, dl, device):
    model.train()

    all_preds = []
    all_labels = []
    all_groups = []
    for iteration, ((x, y, g), name) in enumerate(dl, 0):
        # --------------train------------
        optimizer.zero_grad()
        batch = x.to(device)
        label = y.to(device)

        pred = model(batch)
        loss = criterion(pred, label)      
        loss.backward()
        
        optimizer.step()
        
        all_preds += list(F.softmax(pred, dim=1).cpu().data.numpy())
        all_labels += list(label.cpu().data.numpy())
        all_groups += list(g.cpu().data.numpy())

    # Calculate multiclass AUC
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_groups = np.array(all_groups)
    
    return all_preds, all_labels, all_groups



def validate(model, dl, device):
    all_preds = []
    all_labels = []
    all_names = []
    all_groups = []
    model.eval()
    for iteration, ((x, y, g), name) in enumerate(dl, 0):
        # --------------train------------
        with torch.no_grad():
            batch = x.to(device)
            label = y.to(device)

            pred = model(batch)

            all_preds += list(F.softmax(pred, dim=1).cpu().data.numpy())
            all_labels += list(label.cpu().data.numpy())
            all_names += list(name)
            all_groups += list(g.cpu().data.numpy())

    # Calculate multiclass AUC
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_names = np.array(all_names)
    all_groups = np.array(all_groups)
    return all_preds, all_labels, all_groups, all_names

def evaluate_groups_acc(all_preds, all_labels, all_groups, split="undef"):
    group_accuracies = {}
    n_groups = len(np.unique(all_groups))

    group_dict = {k:[k] for k in range(n_groups)}
    for group in range(n_groups):
        # Indices where the group is equal to the current group in the loop
        
        indices = np.where(np.isin(all_groups, group_dict[group]))
        # Extract predictions and labels for the current group
        group_preds = all_preds[indices]
        group_labels = all_labels[indices]
        
        # Calculate balanced accuracy for the current group
        group_accuracy = balanced_accuracy_score(group_labels, group_preds.argmax(axis=1))
        group_accuracies[split + "_" + str(group)] = group_accuracy

    return group_accuracies

def evaluate_groups_precision(all_preds, all_labels, all_groups, split="undef"):
    group_precisions = {}
    n_groups = len(np.unique(all_groups))
    n_labels = len(np.unique(all_labels))

    for group in range(n_groups):
        for label in range(n_labels):
            # Indices where the group is equal to the current group and the label is equal to the current label
            indices = np.where((all_groups == group) & (all_labels == label))
            if len(indices[0]) > 0:
                # Extract predictions for the current group and label
                group_preds = all_preds[indices]
                # Calculate true positives (TP) and false positives (FP) and precision
                tp = np.sum(group_preds.argmax(axis=1) == label)
                fp = np.sum(group_preds.argmax(axis=1) != label)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                group_precisions[f"{split}_group_{group}_label_{label}"] = precision
            else:
                group_precisions[f"{split}_group_{group}_label_{label}"] = -1.0  # No instances of this label in this group

    return group_precisions

def evaluate_groups_statistics(all_preds, all_labels, all_groups, split="undef"):
    group_precisions = {}
    n_groups = len(np.unique(all_groups))

    for group in range(n_groups):
        # Indices where the group is equal to the current group and the label is equal to the current label
        indices = np.where(all_groups == group)
        if len(indices[0]) > 0:
            # Extract predictions for the current group and label
            group_preds = all_preds[indices]
            # Calculate true positives (TP) and false positives (FP) and precision
            bacc = balanced_accuracy_score(all_labels[indices], group_preds.argmax(axis=1))
            auc = roc_auc_score(all_labels[indices], group_preds[:, 1])

            recall = recall_score(all_labels[indices], group_preds.argmax(axis=1))
            precision = precision_score(all_labels[indices], group_preds.argmax(axis=1))

            group_precisions[f"{split}_group_{group}_recall"] = recall
            group_precisions[f"{split}_group_{group}_precision"] = precision
            group_precisions[f"{split}_group_{group}_bacc"] = bacc
            group_precisions[f"{split}_group_{group}_auc"] = auc

    return group_precisions

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description='Process some integers.')

    # data
    parser.add_argument('--project', help='name of the run')
    parser.add_argument('--bs', type=int, default=64, help='test algo')
    parser.add_argument('--lr', type=float, default=1e-3, help='test algo')
    parser.add_argument('--wd', type=float, default=1e-3, help='test algo')
    parser.add_argument('--wloss', action='store_true')
    parser.add_argument('--degree', type=int, default=2, help='intensity of bias')
    parser.add_argument('--bias_type', type=str, default='hyperintensities', help='artifact type')
    parser.add_argument('--bias_level', type=str, default='0.2', help='artifact type')
    parser.add_argument('--epochs', type=int, default=15, help='epochs')

    args = parser.parse_args()

    project_name = args.project

    wandb.init(project=project_name, entity='X', save_code=True, settings=wandb.Settings(start_method="fork"))

    config = wandb.config
    config.update(args)        

    # Load NIH data

    root = ""
    csv_file = "/storage/homefs/ab24c492/subgroups/data_split/{}_{}_2art_df.csv"
    image_field = "path_to_image"
    target_field = "Cardiomegaly"
    bias_field = ["Contamination_art1","Contamination_art2"]
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
        transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    train_set = CSVDatasetWithName(root, csv_file.format("train", args.bias_level), image_field, target_field, bias_field=bias_field, transform=transform_train)
    val_set = CSVDatasetWithName(root, csv_file.format("val",  args.bias_level), image_field, target_field, bias_field=bias_field, transform=transform_eval)

    #test_set_same = CSVDataset(root, csv_file.format("test", args.bias_level), image_field, target_field, bias_field=bias_field, transform=transform_eval)
    test_set_nobias = CSVDatasetWithName(root, csv_file.format("test", "0"), image_field, target_field, bias_field=bias_field, transform=transform_eval)
    test_set_5050 = CSVDatasetWithName(root, csv_file.format("test", "0.5"), image_field, target_field, bias_field=bias_field, transform=transform_eval)

    
    dl_tr = DataLoader(train_set, batch_size=args.bs, shuffle=True, sampler=None, num_workers=8)
    dl_val = DataLoader(val_set, batch_size=args.bs, shuffle=True, sampler=None, num_workers=8)
    #dl_test_same = DataLoader(test_set_same, batch_size=args.bs, shuffle=True, sampler=None, num_workers=8)
    dl_test_nobias = DataLoader(test_set_nobias, batch_size=args.bs, shuffle=True, sampler=None, num_workers=8)
    dl_test_5050 = DataLoader(test_set_5050, batch_size=args.bs, shuffle=True, sampler=None, num_workers=8)

    # Model
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, len(train_set.class_weights_list))
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.wloss:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(train_set.class_weights_list, dtype=torch.float32).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    results = {}
    best = 0
    for epoch in range(args.epochs):
        tr_preds, tr_labels, tr_groups = train_epoch_ERM(model, optimizer, criterion, dl_tr, device)    
        acc_tr = balanced_accuracy_score(tr_labels, tr_preds.argmax(axis=1))
        auc_tr = roc_auc_score(tr_labels, tr_preds[:, 1])
        #print("[TRAIN] epoch:", epoch, "avg_acc:", acc_tr)
        tr_group_accuracies = evaluate_groups_precision(tr_preds, tr_labels, tr_groups, "train")
        tr_group_statistics = evaluate_groups_statistics(tr_preds, tr_labels, tr_groups, "train")
        tr_group_accuracies['tr_avg_acc'] = acc_tr
        tr_group_accuracies['tr_auc'] = auc_tr
        tr_group_accuracies['epoch'] = epoch
        wandb.log(tr_group_accuracies, commit=True)
        wandb.log(tr_group_statistics, commit=True)

        val_preds, val_labels, val_groups, _ = validate(model, dl_val, device)
        acc_val = balanced_accuracy_score(val_labels, val_preds.argmax(axis=1))
        auc_val = roc_auc_score(val_labels, val_preds[:, 1])
        #print("[VAL] epoch:", epoch, "avg_acc:", acc_val)
        val_group_accuracies = evaluate_groups_precision(val_preds, val_labels, val_groups, "val")
        val_group_statistics = evaluate_groups_statistics(val_preds, val_labels, val_groups, "val")
        val_group_accuracies['val_avg_acc'] = acc_val
        val_group_accuracies['val_auc'] = auc_val
        val_group_accuracies['epoch'] = epoch
        wandb.log(val_group_accuracies, commit=True)
        wandb.log(val_group_statistics, commit=True)

        test_preds_nobias, test_labels_nobias, test_groups_nobias, _ = validate(model, dl_test_nobias, device)
        acc_test_nobias = balanced_accuracy_score(test_labels_nobias, test_preds_nobias.argmax(axis=1))
        auc_test_nobias = roc_auc_score(test_labels_nobias, test_preds_nobias[:, 1])
        #print("[TEST NOBIAS] epoch:", epoch, "avg_acc:", acc_test_nobias)
        test_group_accuracies_nobias = evaluate_groups_precision(test_preds_nobias, test_labels_nobias, test_groups_nobias, "test_nobias")
        test_group_statistics_nobias = evaluate_groups_statistics(test_preds_nobias, test_labels_nobias, test_groups_nobias, "test_nobias")
        test_group_accuracies_nobias['test_nobias_avg_acc'] = acc_test_nobias
        test_group_accuracies_nobias['test_nobias_auc'] = auc_test_nobias
        test_group_accuracies_nobias['epoch'] = epoch
        wandb.log(test_group_accuracies_nobias, commit=True)
        wandb.log(test_group_statistics_nobias, commit=True)

        test_preds_5050, test_labels_5050, test_groups_5050, _ = validate(model, dl_test_5050, device)
        acc_test_5050 = balanced_accuracy_score(test_labels_5050, test_preds_5050.argmax(axis=1))
        auc_test_5050 = roc_auc_score(test_labels_5050, test_preds_5050[:, 1])
        #print("[TEST 5050] epoch:", epoch, "avg_acc:", acc_test_5050)
        test_group_accuracies_5050 = evaluate_groups_precision(test_preds_5050, test_labels_5050, test_groups_5050, "test_5050")
        test_group_statistics_5050 = evaluate_groups_statistics(test_preds_5050, test_labels_5050, test_groups_5050, "test_5050")
        test_group_accuracies_5050['test_5050_avg_acc'] = acc_test_5050
        test_group_accuracies_5050['test_5050_auc'] = auc_test_5050
        test_group_accuracies_5050['epoch'] = epoch
        wandb.log(test_group_accuracies_5050, commit=True)
        wandb.log(test_group_statistics_5050, commit=True)

        #worst_val_acc = min(val_group_accuracies.values())
        if acc_val >= best:
            best = acc_val
            best_group = copy.deepcopy(val_group_accuracies)
            best_group['epoch'] = epoch

            save_checkpoint(model, epoch, project_name + '_' + wandb.run.name)

    
    # Load the best model according to validation, and capture performances of val and test.
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 2)
    
    name = project_name + '_' + wandb.run.name
    model_path = "../../saves/" + name + ".pth"
    load_model(model, model_path, torch.device("cuda"))
    model.to(device)
    
    # Create a new artifact
    artifact = wandb.Artifact(
        name='model_predictions',    # Name of the artifact
        type='dataset',              # Type can be 'dataset', 'model', etc.
        description='Predictions from the model for the test dataset'  # Optional description
    )

    val_preds, val_labels, val_group, val_names = validate(model, dl_val, device)
    test_preds, test_labels, test_group, test_names = validate(model, dl_test_5050, device)
    
    num_dims = val_preds.shape[1]  # e.g., if test_preds is (n_samples, 3), num_dims = 3
    pred_columns = [f'pred_{i}' for i in range(num_dims)]

    df_test_preds = pd.DataFrame(test_preds, columns=pred_columns)
    df_test_preds['name'] = test_names
    df_test_preds['gt'] = test_labels
    df_test_preds['group'] = test_group
    df_test_preds = df_test_preds[['name'] + ['gt'] + ['group'] + pred_columns]

    val_pred_columns = [f'pred_{i}' for i in range(num_dims)]
    df_val_preds = pd.DataFrame(val_preds, columns=pred_columns)
    df_val_preds['name'] = val_names
    df_val_preds['gt'] = val_labels
    df_val_preds['group'] = val_group
    df_val_preds = df_val_preds[['name'] + ['gt'] + ['group'] + pred_columns]

    # Save the DataFrame to CSV
    df_val_preds.to_csv("../../saves/" + name + "_valpreds.csv", index=False)
    df_test_preds.to_csv("../../saves/" + name + "_testpreds.csv", index=False)

    # Add the CSV file to the artifact
    artifact.add_file("../../saves/" + name + "_testpreds.csv")
    artifact.add_file("../../saves/" + name + "_valpreds.csv")

    # Log the artifact to wandb
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
