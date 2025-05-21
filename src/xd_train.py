import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import xd_option




def one_hot_to_descriptive_text(one_hot_encoded_labels, index_to_label):
    text_labels = []
    for one_hot in one_hot_encoded_labels:
        indices = torch.where(one_hot == 1)[0].tolist()
        labels = [index_to_label[idx] for idx in indices]
        text_label = ' '.join(labels)
        text_labels.append(text_label)
    return text_labels

def compute_ranking_loss(S_n_n, S_n_a, S_a_n, S_a_a_org, S_a_a,normal_lengths,abnormal_lengths):
    """
    Compute the ranking loss for normal and anomalous videos.
    
    Args:
    S_n_n (torch.Tensor): Similarities between normal video features and normal event description text embedding.
    S_n_a (torch.Tensor): Similarities between normal frames and the description texts of other anomalous events.
    S_a_n (torch.Tensor): Similarities between anomalous video features and normal event description text embedding.
    S_a_a_org (torch.Tensor): Similarities between anomalous video features and the description text embedding of the real anomalous event.
    S_a_a (torch.Tensor): Similarity set between anomalous video features and description text embedding of other anomalous event categories.
    
    Returns:
    L_rank_n (torch.Tensor): Ranking loss for normal videos.
    L_rank_a (torch.Tensor): Ranking loss for anomalous videos.
    """
    s_nn,s_na,s_an,saa_org,s_aa=torch.zeros(0).to(device),torch.zeros(0).to(device),torch.zeros(0).to(device),torch.zeros(0).to(device),torch.zeros(0).to(device)
    for i in range(S_n_n.shape[0]):

        tmp_=S_n_n[i, 0:normal_lengths[i]]
        s_nn=torch.cat([s_nn, tmp_], dim=0)
        tmp_2=S_n_a[i,0:normal_lengths[i]]
        s_na=torch.cat([s_na, tmp_2], dim=0)

    for i in range(S_a_a.shape[0]):

        tmp_=S_a_n[i, 0:abnormal_lengths[i]]
        s_an=torch.cat([s_an, tmp_], dim=0)
        tmp_2=S_a_a_org[i,0:abnormal_lengths[i]]
        saa_org=torch.cat([saa_org, tmp_2], dim=0)
        tmp_3=S_a_a[i, 0:abnormal_lengths[i]]
        s_aa=torch.cat([s_aa, tmp_3], dim=0)  

    # Compute the ranking loss for normal videos
    max_S_n_n = torch.max(s_nn)
    max_S_n_a = torch.max(s_na)

    L_rank_n = torch.max(torch.tensor(0.0), torch.tensor(1.0) - max_S_n_n + max_S_n_a) 
    # Compute the ranking loss for anomalous videos
    max_S_a_n = torch.max(s_an)
    max_S_a_a_org = torch.max(saa_org)
    max_S_a_a = torch.max(s_aa)

    term1 = torch.max(torch.tensor(0.0), torch.tensor(1.0) - max_S_a_n + max_S_a_a)
    term2 = torch.max(torch.tensor(0.0), torch.tensor(1.0) - max_S_a_a_org + max_S_a_a)
    L_rank_a = term1 + term2
    
    return L_rank_n, L_rank_a


def train(model, train_loader, test_loader, args, label_map: dict, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)#real class names
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        # loss_total1 = 0
        # loss_total2 = 0
        loss_total4=0
        loss_total5=0

        loss_rank_n=0 
        loss_rank_a=0
        for i, item in enumerate(train_loader):
            step = 0
            visual_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            
            # Create reverse map from prompt_text index to label_map values
            index_to_label = {prompt_text.index(value): value for key, value in label_map.items()}
 
            one_hot_encoded_labels=text_labels
            # Get descriptive text labels from one-hot encoded tensor
            text_labels2 = one_hot_to_descriptive_text(one_hot_encoded_labels, index_to_label)

            #___________________
            text_features, logits1, logits2,_,_,realanomalyembeddings,normal_enhanced_embeddings,all_S_n_n,all_S_n_a,normal_lengths = model(visual_feat, None, prompt_text, feat_lengths,text_labels2)       
            # Initialize tensors to store the values
            text_feature_abr_real = torch.zeros(0).to(device)
            visual_features_abr = torch.zeros(0).to(device)
            num_zeros=0
            # Iterate through the text labels and concatenate the values to the corresponding tensors
            # Initialize an empty list to store the indices
            normal_indices = []
            abnormal_indices =[]

            # Labels to exclude
            exclude_labels = ['fighting', 'shooting', 'riot', 'abuse', 'car accident', 'explosion']
            otherencodedtextlabels_abr_ =torch.zeros(0).to(device)
            normalized_text_feature_abr=torch.zeros(0).to(device)
            # Maximum number of labels to exclude
            max_exclude_labels = len(exclude_labels)
            abnormal_lengths=[]

            # Loop through the labels and check for 'normal' labels
            for j, label in enumerate(text_labels2):
                if label == 'normal':
                    normal_indices.append(j)
                    num_zeros += 1

                else:
                    abnormal_lengths.append(feat_lengths[j])
                    normalized_text_feature_abr = torch.cat((normalized_text_feature_abr,realanomalyembeddings[j].unsqueeze(0)), dim=0)
                    normalized_visual_feature_abr = visual_feat[j]
                    current_textlabel=text_labels2[j]
                    if any(lbl in exclude_labels for lbl in current_textlabel.split()) or  current_textlabel in exclude_labels:
 
                        otherlabels = [l for l in exclude_labels if l not in current_textlabel]

                        # Encode the remaining labels
                        otherencodedtextlabels_abr = model.encode_textprompt(otherlabels)

                        if otherencodedtextlabels_abr.shape[0] < max_exclude_labels - 1:
                            padding = torch.zeros((max_exclude_labels - 1 - otherencodedtextlabels_abr.shape[0], 512)).to(device)
                            otherencodedtextlabels_abr = torch.cat((otherencodedtextlabels_abr, padding), dim=0)
                                              
                        otherencodedtextlabels_abr_ = torch.cat((otherencodedtextlabels_abr_, otherencodedtextlabels_abr.unsqueeze(0)), dim=0)

                    text_feature_abr_real=normalized_text_feature_abr
                    visual_features_abr = torch.cat((visual_features_abr, normalized_visual_feature_abr.unsqueeze(0)), dim=0)
                    abnormal_indices.append(j)

            otherencodedtextlabels_abr_=otherencodedtextlabels_abr_.permute(0, 2, 1)
            text_feature_abr_real=text_feature_abr_real.unsqueeze(-1)

            S_a_a=visual_features_abr @otherencodedtextlabels_abr_.type(visual_features_abr.dtype)
            S_a_n=normalityTextual_anomalyVisualfeatures_alignment=visual_features_abr @ normal_enhanced_embeddings.type(visual_features_abr.dtype) #/ 0.07 
            S_a_a_org=anomalyTextual_anomalyVisualfeatures_alignment=visual_features_abr @ text_feature_abr_real.type(visual_features_abr.dtype) #/ 0.07 
        
            normalityTextual_anomalyVisualfeatures_alignment = normalityTextual_anomalyVisualfeatures_alignment / (normalityTextual_anomalyVisualfeatures_alignment.norm(dim=-1, keepdim=True)+ 1e-8)
            anomalyTextual_anomalyVisualfeatures_alignment = anomalyTextual_anomalyVisualfeatures_alignment / (anomalyTextual_anomalyVisualfeatures_alignment.norm(dim=-1, keepdim=True) +  1e-8)
           
            # Define the guidance weight α
            alpha = 0.2  # You can choose a different value if needed
            psi_i = alpha * anomalyTextual_anomalyVisualfeatures_alignment+ (1 - alpha) * (1 - normalityTextual_anomalyVisualfeatures_alignment.unsqueeze(-1) )
        
            # Define a threshold 
            theta = 0.81 # You can choose a different value if needed

            #_____________________Obtain pseudo-labels  with the mean threshold ______________________
            gamma_a_i_j = (psi_i >= theta).float()
            zeros_tensor = torch.zeros((len(text_labels2), gamma_a_i_j.shape[1], gamma_a_i_j.shape[2]), device=gamma_a_i_j.device)

            # Assign gamma_a_i_j values to the abnormal indices
            for n,idx in enumerate(abnormal_indices):
                zeros_tensor[idx] = gamma_a_i_j[n]
            

            pseudo_labels=zeros_tensor

            Logits1 = torch.zeros(0).to(device) #
            Pseudo_labels = torch.zeros(0).to(device)
            lengths=feat_lengths
            Ph_i_lenghts =torch.zeros(0).to(device)
            normal_label_len=0
            for k in range (len(text_labels2)):
                if text_labels2[k]=='normal':
                    normal_label_len+=1

            icount=0
            for p in range(pseudo_labels.shape[0]):
                tmp_=logits1[p, 0:lengths[p]]
                Logits1=torch.cat([Logits1,tmp_ ],dim=0)
                tmp_2= pseudo_labels[p, 0:lengths[p]]
                Pseudo_labels=torch.cat([Pseudo_labels,tmp_2 ],dim=0)
                icount+=1
                if text_labels2[p]!='normal':
                    # print("lengths[p]",lengths[p])
                    tmp_3= psi_i[p-icount, 0:lengths[p]]
                    Ph_i_lenghts = torch.cat([Ph_i_lenghts, tmp_3], dim=0)
            all_anomaly_labels=[]
            start_idx = 0  # Start index for slicing
            expectations=[]

            for length in abnormal_lengths:
                end_idx = start_idx + length  # End index for slicing
                mean_value = torch.mean(Ph_i_lenghts[start_idx:end_idx])
                expectations.append(mean_value)

                # Update the start index for the next slice
                start_idx = end_idx

            PL_tensor2 = torch.zeros(0).to(device)  # Initialize an empty tensor on the specified device
            start2 = 0  # To keep track of the starting index for anomaly labels

            for x, lbl in enumerate(text_labels2):
                feat_length = feat_lengths[x]  # Length of current feature set

                if lbl == 'normal':
                    # Append zeros for the length of the current feature
                    zeros_to_append = torch.zeros(feat_length).to(device)
                    PL_tensor2 = torch.cat((PL_tensor2, zeros_to_append), dim=0)
                
                else:  # lbl != 'Normal'
                    # Append the corresponding anomaly labels
                    end = start2 + feat_length
                    anomaly_labels_to_append = all_anomaly_labels[start2:end]
                    anomaly_labels_to_append = torch.tensor(anomaly_labels_to_append).to(device)

                    PL_tensor2 = torch.cat((PL_tensor2, anomaly_labels_to_append), dim=0)

                    # Update the start position for the next iteration
                    start2 = end

            gamma = torch.zeros(0, psi_i.shape[1], psi_i.shape[2]).to(device)
            for exp in range(len(expectations)):
                # Apply the comparison and concatenate the result to gamma
                gamma = torch.cat([gamma, (psi_i[exp] >= expectations[exp]).float().unsqueeze(0)], dim=0)

            zeros_tensor3 = torch.zeros((len(text_labels2), gamma.shape[1], gamma.shape[2]), device=gamma.device)

            # Assign gamma_a_i_j values to the abnormal indices
            for ns,idxs in enumerate(abnormal_indices):
                zeros_tensor3[idxs] = gamma[ns]

            Pseudo_labels3 = torch.zeros(0).to(device)
            
            for m in range(zeros_tensor3.shape[0]):
                tmp_33=zeros_tensor3[m, 0:lengths[m]]
                Pseudo_labels3=torch.cat([Pseudo_labels3,tmp_33 ],dim=0)        


            loss_4_BCE = F.binary_cross_entropy_with_logits( Logits1,Pseudo_labels3) # with the mean threshold 

            loss_total4 += loss_4_BCE.item()

#______________________ Getting the categorical pseudo labels for CE loss____________
            # Initialize Categorical_pseudo_labels with the same shape as logits2
            Categorical_pseudo_labels = torch.zeros(logits2.shape, dtype=torch.float32, device='cuda')
            PL_tensor2 = PL_tensor2.unsqueeze(dim=1)  # Add an extra dimension to make it (X, 1)


            default_tensor = torch.tensor([1., 0., 0., 0., 0., 0., 0.])

            offset = 0

            # Loop over each video's frame length
            for l in range(len(feat_lengths)):  # Iterate over each video label
                current_length = feat_lengths[l] 
                for pp in range(current_length):
       
                    if Pseudo_labels3[offset + pp] == 0:
                        Categorical_pseudo_labels[l, pp] = default_tensor
                         
                    else:
                        Categorical_pseudo_labels[l, pp] = text_labels[l]
                offset += current_length

            Logits2 = torch.zeros(0).to(device)
            Pseudo_labels_cat = torch.zeros(0).to(device)
            for p in range(logits2.shape[0]):
                tmp_=logits2[p, 0:lengths[p]]
                Logits2=torch.cat([Logits2,tmp_] ,dim=0)
                tmp_2= Categorical_pseudo_labels[p, 0:lengths[p]]
                Pseudo_labels_cat=torch.cat([Pseudo_labels_cat,tmp_2 ],dim=0)


            # Reshape logits and labels for cross-entropy loss
            Logits2 = Logits2.view(-1, logits2.shape[2])  # Reshape to (X*256, 14)
            Pseudo_labels_cat = Pseudo_labels_cat.view(-1,Categorical_pseudo_labels.shape[2])     #  Reshape to (X*256, 14)
            loss5_CE= F.binary_cross_entropy_with_logits(Logits2, Pseudo_labels_cat)
            loss_total5 += loss5_CE.item()

            L_rank_n, L_rank_a = compute_ranking_loss(all_S_n_n, all_S_n_a, S_a_n, S_a_a_org, S_a_a,normal_lengths,abnormal_lengths)

            loss_rank_n+=L_rank_n.item()
            loss_rank_a+=L_rank_a.item()
            loss3 = torch.zeros(1).to(device)

            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6



            loss=loss_4_BCE + loss5_CE + loss3* 1e-4+L_rank_n+L_rank_a

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * train_loader.batch_size
            if step % 4800 == 0 and step != 0:

                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total4 / (i+1), '| loss2: ', loss_total5 / (i+1), '| loss3: ', loss3.item(),'| loss_rank_n: ', loss_rank_n / (i+1), '| loss_rank_a: ', loss_rank_a / (i+1))

        scheduler.step()
        AUC, AP, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
        #AP = mAP

        if AP > ap_best:
            ap_best = AP 
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, args.checkpoint_path)

        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    train(model, train_loader, test_loader, args, label_map, device)


    