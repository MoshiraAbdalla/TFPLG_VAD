import numpy as np
import torch

from utils.dataset import XDDataset
import time
import MSAD_option
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from ucf_test import test


import pandas as pd
import numpy as np


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
 
    video_times = []
    model.to(device)
    model.eval()

    element_logits2_stack = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]
 
            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)
 
            lengths = torch.zeros(int(length / maxlen) + 1)
            


            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            artificial_labels=["normal"]
            artificial_labels= np.repeat( artificial_labels,visual.size(0))
            start_time = time.time()

            _, logits1, logits2,_,_,_,_,_,_,_ = model(visual, padding_mask, prompt_text, lengths,artificial_labels)
            end_time = time.time()
            # print(f"[Video {i}] Inference time: {end_time - start_time:.4f} seconds")
            video_times.append(end_time - start_time) 
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2

            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()

            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()
    avg_time = sum(video_times) / len(video_times)
    # print(f"\n Average inference time per video: {avg_time:.4f} seconds")
  

    gt = np.pad(gt, (0, len(ap1) - len(gt)), mode='constant', constant_values=0)

    total_time = sum(video_times)

    avg_time_per_frame = total_time / len(gt)
    print(f" Average inference time per frame: {avg_time_per_frame * 1000:.4f} ms")    
    ROC1 = roc_auc_score(gt, ap1)
    AP1 = average_precision_score(gt, ap1)
 

    print("AUC1: ", ROC1, " AP1: ", AP1)
  





  

    return ROC1, AP1


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args=MSAD_option.parser.parse_args()
    label_map = dict({'Assault': 'Assault',   'Explosion': 'Explosion', 'Fighting': 'Fighting', 'Fire':'Fire',  'Object_falling' : 'Object_falling','People_falling':'People_falling','Robbery':'Robbery','Shooting':'Shooting','Traffic_accident':'Traffic_accident','Vandalism':'Vandalism','Water_incident':'Water_incident','Normal': 'Normal'})
    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    testdataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    # testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt_df = pd.read_excel("./list/final_ground_truth.xlsx", engine='openpyxl') # this is only to match the abnormal from the i3d 

    gt = gt_df.values.flatten()


    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)


    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)


    test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
