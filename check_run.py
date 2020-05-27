import numpy as np
import torch
import argparse
import os
from os import path
from sklearn import metrics
from metrics import trackers
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='folder')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    model_exists = path.exists(args.f + "/model.pt")
    if not model_exists:
        print(args.f + " FAILED TRAINING\n")
    preds_exist = path.exists(args.f + "/out_test.npy")
    if not preds_exist:
        print(args.f + " FAILED INFERENCE\n")
    if model_exists:
        history = torch.load(args.f + "/model.pt")["history"]
        success = len(history.train_loss) == 50
        if not success:
            print(args.f + " FAILED - INCOMPLETE TRAINING\n")
        if success and preds_exist:
            print(args.f + " SUCCESS")
            with open(args.f + "/log_test.txt") as f:
                content = f.readlines()

            # print pearson score and mae
            print(content[15])
            print(content[14])
                
