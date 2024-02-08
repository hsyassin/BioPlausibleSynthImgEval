import numpy as np
import os

import torch
import re
from textwrap import wrap
import matplotlib.pyplot as plt



def beautify_labels(labels):
    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]
    return classes


def dice_score_perclass(vol_output, ground_truth, num_classes):
    dice_perclass = torch.zeros(num_classes)

    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        smooth = 1
        GT = torch.flatten(GT)
        Pred = torch.flatten(Pred)
        intersection = torch.sum(GT * Pred)
        union = (torch.sum(GT) + torch.sum(Pred) + smooth)
        dice_perclass[i] =  (2. * intersection + smooth) / union

    return dice_perclass

# def dice_score_perclass(vol_output, ground_truth, num_classes):
#     dice_perclass = np.zeros(num_classes)

#     for i in range(num_classes):
#         GT = (ground_truth == i).astype(float)
#         Pred = (vol_output == i).astype(float)
#         smooth = 1
#         GT = GT.flatten()
#         Pred = Pred.flatten()
#         intersection = np.sum(GT * Pred)
#         union = (np.sum(GT) + np.sum(Pred) + smooth)
#         dice_perclass[i] =  (2. * intersection + smooth) / union

#     return dice_perclass

def plot_dice_score(ds, age, sex, num_class=3, title=""):
    labels = ["BKG", "LLV", "RLV"]
    labels = beautify_labels(labels)
    fig = plt.figure(figsize=(8, 6), dpi=180, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.axes.set_title(title, fontsize=10)                   #added by HY
    extra = "_Avg" if "Average" in title else ""
    ax.set_ylabel("Dice Score", fontsize=7)                 #added by HY
    ax.yaxis.set_label_position('left')                     #added by HY
    #perclass
    ax.set_xlabel("Brain Regions", fontsize=7)          #added by HY
    ax.bar(np.arange(num_class), ds)               
    ax.set_xticks(np.arange(num_class))
    c = ax.set_xticklabels(labels, fontsize=6, rotation=-90, ha='center')
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position('bottom')
    plt.show()
    os.makedirs(f'DSC_per_class/', exist_ok=True)
    plt.savefig(f'DSC_per_class/Age-{age}_Sex-{sex}{extra}.png' , format="png")
    plt.close()


def dice_confusion_matrix(vol_output, ground_truth, num_classes):
    dice_cm = torch.zeros(num_classes, num_classes)
    vol_output, ground_truth = torch.from_numpy(vol_output), torch.from_numpy(ground_truth)

    for i in range(num_classes):
        GT = (ground_truth == i).float()
        for j in range(num_classes):
            Pred = (vol_output == j).float()

            smooth = 1
            GT = torch.flatten(GT)
            Pred = torch.flatten(Pred)
            intersection = torch.sum(GT * Pred)
            union = (torch.sum(GT) + torch.sum(Pred) + smooth)
            dice_cm[i, j] = (2. * intersection + smooth) / union
    avg_dice = torch.mean(torch.diagflat(dice_cm))
    return avg_dice, dice_cm


def plot_cm(cm, age, sex, num_class=3, title=""):
    cm_cmap=plt.cm.Blues
    labels = ["BKG", "LLV", "RLV"]
    labels = beautify_labels(labels)
    fig = plt.figure(figsize=(8, 8), dpi=180, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(cm, interpolation='nearest', cmap=cm_cmap)
    ax.axes.set_title(title, fontsize=10)                   #added by HY
    extra = "Avg" if "Average" in title else ""
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(np.arange(num_class))
    c = ax.set_xticklabels(labels, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(np.arange(num_class))
    ax.set_yticklabels(labels, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], '.4f') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=6,
                verticalalignment='center', color="white" if cm[i, j] > thresh else "black")

    fig.set_tight_layout(True)
    np.set_printoptions(precision=2)
    plt.show()
    os.makedirs(f'DSC_cm_per_class/', exist_ok=True)
    plt.savefig(f'DSC_cm_per_class/Age-{age}_Sex-{sex}{extra}.png' , format="png")
    plt.close()
