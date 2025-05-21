import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import matplotlib.pyplot as plt

from model import CLIPVAD
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option
from matplotlib.animation import FuncAnimation

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    labels_=torch.zeros(0).to(device)
    lengths_=torch.zeros(0).to(device)
    for j in range(labels.shape[0]):
        if labels[j,0]==1:
            continue
        else:
            labels_=torch.cat([labels_,labels[j].unsqueeze(0)],dim=0)
            lengths_=torch.cat([lengths_,lengths[j].unsqueeze(0)],dim=0)

    labels_ = labels_[:, 1:]
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:int(lengths_[i].item())], k=int(int(lengths_[i].item()) / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(labels_ * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss,labels_

#The CLAS2 function calculates a classification loss for binary classification. 
def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    #For each instance in the batch, select the top-k values and compute their mean.
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

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
    #print("max_S_n_n",max_S_n_n)
    max_S_n_a = torch.max(s_na)
    #print("max_S_n_a",max_S_n_a)

    L_rank_n = torch.max(torch.tensor(0.0), torch.tensor(1.0) - max_S_n_n + max_S_n_a)
    
    # Compute the ranking loss for anomalous videos
    max_S_a_n = torch.max(s_an)
    max_S_a_a_org = torch.max(saa_org)
    max_S_a_a = torch.max(s_aa)


    term1 = torch.max(torch.tensor(0.0), torch.tensor(1.0) - max_S_a_n + max_S_a_a)
    term2 = torch.max(torch.tensor(0.0), torch.tensor(1.0) - max_S_a_a_org + max_S_a_a)
    L_rank_a = term1 + term2
    
    return L_rank_n, L_rank_a


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
 
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
        loss_total1 = 0
        loss_total2 = 0
        loss_total4=0
        loss_total5=0
        loss_rank_n=0 
        loss_rank_a=0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter) # to fetch the next batch
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device) # uses both features from normal and abnormal 
            text_labels = list(normal_label) + list(anomaly_label)
            text_labels2 = list(normal_label) + list(anomaly_label) #real anomaly labels

            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)



            text_features, logits1, logits2,logit2,text_features_aftervisualprompt,realanomalyembeddings,normal_enhanced_embeddings,all_S_n_n,all_S_n_a,normal_lengths = model(visual_features, None, prompt_text, feat_lengths,text_labels2) 

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
 
            loss2,labels_cat = CLASM(logit2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            #This loss is a regularization term to enforce similarity between the first text feature (assumed to be a normal class) and other text features.
            loss3 = torch.zeros(1).to(device)

            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)

            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True) # from orginal textual descriptions
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            #Average the accumulated similarity losses and scale.
            # print("loss3 before",loss3)
            loss3 = loss3 / 13 * 1e-1



            # Initialize tensors to store the values
            text_feature_abr_real = torch.zeros(0).to(device)
            visual_features_abr = torch.zeros(0).to(device)
            num_zeros=0

            normal_indices = []
            abnormal_indices =[]

            # Labels to exclude :Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'
            exclude_labels = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary','Explosion','Fighting','RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism']
            otherencodedtextlabels_abr_ =torch.zeros(0).to(device)
            normalized_text_feature_abr=torch.zeros(0).to(device)
            # Maximum number of labels to exclude
            max_exclude_labels = len(exclude_labels)
            abnormal_lengths=[]
            # Loop through the labels and check for 'normal' labels
            for j, label in enumerate(text_labels2):
                if label == 'Normal':
                    normal_indices.append(j)
                    num_zeros += 1

                else:
                    abnormal_lengths.append(feat_lengths[j])
                    normalized_text_feature_abr = torch.cat((normalized_text_feature_abr,realanomalyembeddings[j].unsqueeze(0)), dim=0)#/ realanomalyembeddings[j].norm(dim=-1, keepdim=True).unsqueeze(1)  
                    normalized_visual_feature_abr = visual_features[j]# / visual_feat[j].norm(dim=-1, keepdim=True) # anomaly visual features from clip
                    current_textlabel=text_labels2[j]

                    # Check if the current label is one of the specific labels

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

            S_a_n=normalityTextual_anomalyVisualfeatures_alignment=visual_features_abr @ normal_enhanced_embeddings.type(visual_features_abr.dtype) #/ 0.07# Alignment map between text features and visual features
            S_a_a_org=anomalyTextual_anomalyVisualfeatures_alignment=visual_features_abr @ text_feature_abr_real.type(visual_features_abr.dtype) #/ 0.07# Alignment map between text features and visual features
          
            normalityTextual_anomalyVisualfeatures_alignment = normalityTextual_anomalyVisualfeatures_alignment / (normalityTextual_anomalyVisualfeatures_alignment.norm(dim=-1, keepdim=True)+ 1e-8)
            anomalyTextual_anomalyVisualfeatures_alignment = anomalyTextual_anomalyVisualfeatures_alignment / (anomalyTextual_anomalyVisualfeatures_alignment.norm(dim=-1, keepdim=True) +  1e-8)
            
            alpha = 0.2  # You can choose a different value if needed

            psi_i = alpha * anomalyTextual_anomalyVisualfeatures_alignment+ (1 - alpha) * (1 - normalityTextual_anomalyVisualfeatures_alignment.unsqueeze(-1) )
            # Define a threshold 
            theta = 0.81 # You can choose a different value if needed
            # Obtain pseudo-labels
            gamma_a_i_j = (psi_i >= theta).float()

            # Initialize a new tensor for the result
            zeros_tensor = torch.zeros((len(text_labels2), gamma_a_i_j.shape[1], gamma_a_i_j.shape[2]), device=gamma_a_i_j.device)
            # Assign gamma_a_i_j values to the abnormal indices
            for n,idx in enumerate(abnormal_indices):
                zeros_tensor[idx] = gamma_a_i_j[n]

            pseudo_labels=zeros_tensor

            Logits1 = torch.zeros(0).to(device) #
            Pseudo_labels = torch.zeros(0).to(device)
            lengths=feat_lengths
            Ph_i_lenghts =torch.zeros(0).to(device)
            for p in range(pseudo_labels.shape[0]):
                tmp_=logits1[p, 0:lengths[p]]
                Logits1=torch.cat([Logits1,tmp_ ],dim=0)
                tmp_2= pseudo_labels[p, 0:lengths[p]]
                Pseudo_labels=torch.cat([Pseudo_labels,tmp_2 ],dim=0)
                if text_labels2[p]!='Normal':
                    # print("lengths[p]",lengths[p])
                    tmp_3= psi_i[p-len(normal_label), 0:lengths[p]]
                    Ph_i_lenghts = torch.cat([Ph_i_lenghts, tmp_3], dim=0)

#___________ Computing the binary cross-entropy loss for pseudo labels using 80th percentile Threshold___________
            all_anomaly_labels=[]
            start_idx = 0  # Start index for slicing
            c=0
            for length in anomaly_lengths:
                end_idx = start_idx + length  # End index for slicing
                values = Ph_i_lenghts[start_idx:end_idx].detach().cpu().numpy()

                # Set the threshold as the 80th percentile
                percentile_threshold = np.percentile(values, 80)
                c+=1
                anomaly_labels = torch.tensor(values >= percentile_threshold).float()

                all_anomaly_labels.extend(anomaly_labels)
                start_idx = end_idx

            PL_tensor2 = torch.zeros(0).to(device)  
            start2 = 0  
            for x, lbl in enumerate(text_labels2):
                feat_length = feat_lengths[x]  
               
                if lbl == 'Normal':
                    zeros_to_append = torch.zeros(feat_length).to(device)
                    PL_tensor2 = torch.cat((PL_tensor2, zeros_to_append), dim=0)               
                else:  # lbl != 'Normal'
                    # Append the corresponding anomaly labels
                    end = start2 + feat_length
                    anomaly_labels_to_append = all_anomaly_labels[start2:end]
                    anomaly_labels_to_append = torch.tensor(anomaly_labels_to_append).to(device)
                    PL_tensor2 = torch.cat((PL_tensor2, anomaly_labels_to_append), dim=0)
                    start2 = end          
            Logits1 = Logits1.squeeze(dim=1)
            loss_4_BCE = F.binary_cross_entropy_with_logits( Logits1,PL_tensor2)
            loss_total4 += loss_4_BCE.item()
            L_rank_n, L_rank_a = compute_ranking_loss(all_S_n_n, all_S_n_a, S_a_n, S_a_a_org, S_a_a,normal_lengths,abnormal_lengths) 
            loss_rank_n+=L_rank_n.item()
            loss_rank_a+=L_rank_a.item()          
            PL_tensor2 = PL_tensor2.unsqueeze(dim=1)         
            loss = loss_4_BCE + loss2 + loss3+L_rank_n+L_rank_a


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            step += i * normal_loader.batch_size * 2
            #print("step",step)
            if step % 1280 == 0 and step != 0:
                #print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total4 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item(),'| loss_rank_n: ', loss_rank_n / (i+1), '| loss_rank_a: ', loss_rank_a / (i+1))
   
                AUC, AP,averageMAP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                AP = AUC

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                
        scheduler.step()
        
        torch.save(model.state_dict(), 'model/model_cur.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    

    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)