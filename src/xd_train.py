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

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

# def CLASM(logits, labels, lengths, device):
#     instance_logits = torch.zeros(0).to(device)
#     #print("labels.shape",labels.shape)
#     #print("labels",labels)
#     labels = labels / torch.sum(labels, dim=1, keepdim=True)
#     labels = labels.to(device)
#     labels_=torch.zeros(0).to(device)
#     lengths_=torch.zeros(0).to(device)
#     for j in range(labels.shape[0]):
#         if labels[j,0]==1:
#             continue
#         else:
#             labels_=torch.cat([labels_,labels[j].unsqueeze(0)],dim=0)
#             lengths_=torch.cat([lengths_,lengths[j].unsqueeze(0)],dim=0)

#     labels_ = labels_[:, 1:]

#     #print("labels_.shape",labels_.shape) # (X,14)
#     #print("labels_",labels_)
#     #print("lengths_",lengths)
#     #print("lengths_",lengths_)
#     for i in range(logits.shape[0]):
#         tmp, _ = torch.topk(logits[i, 0:int(lengths_[i].item())], k=int(int(lengths_[i].item()) / 16 + 1), largest=True, dim=0)
#         instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
#     #print("instance_logits.shape",instance_logits.shape) # torch.Size([64, 14])
#     milloss = -torch.mean(torch.sum(labels_ * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
#     return milloss,labels_

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss
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
    #print("S_n_n, S_n_a, S_a_n, S_a_a_org, S_a_a shapes",S_n_n.shape, S_n_a.shape, S_a_n.shape, S_a_a_org.shape, S_a_a.shape) # ([32, 256]) torch.Size([32, 256, 13]) torch.Size([32, 256]) torch.Size([32, 256, 1]) torch.Size([32, 256, 12]
    s_nn,s_na,s_an,saa_org,s_aa=torch.zeros(0).to(device),torch.zeros(0).to(device),torch.zeros(0).to(device),torch.zeros(0).to(device),torch.zeros(0).to(device)
    #print("noraml lengths",torch.sum(torch.stack(normal_lengths)))
    #print("abnormal_lengths ",torch.sum(torch.stack(abnormal_lengths)))

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
    #print("s_nn.shape, s_na.shape,s_an,saa_org,s_aa",s_nn.shape, s_na.shape,s_an.shape,saa_org.shape,s_aa.shape) 
    

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
    #print("max_S_a_n",max_S_a_n)
    #print("max_S_a_a_org",max_S_a_a_org)
    #print("max_S_a_a",max_S_a_a)

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
        loss_total1 = 0
        loss_total2 = 0
        loss_total4=0
        loss_total5=0

        loss_rank_n=0 
        loss_rank_a=0
        for i, item in enumerate(train_loader):
            step = 0
            visual_feat, text_labels, feat_lengths = item
            #print("feat_lengths",feat_lengths)
            #print("visual_feat",visual_feat)


            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            #print("text_labels",text_labels)#one hot encoded
            
            #-------------------my addition------------
            # Create reverse map from prompt_text index to label_map values
            index_to_label = {prompt_text.index(value): value for key, value in label_map.items()}
            #print("index_to_label",index_to_label)

            one_hot_encoded_labels=text_labels
            # Get descriptive text labels from one-hot encoded tensor
            text_labels2 = one_hot_to_descriptive_text(one_hot_encoded_labels, index_to_label)
            # print(text_labels2)
            # Print the descriptive text labels
            #for one_hot, text_label in zip(one_hot_encoded_labels, text_labels2):
                #print(f"{one_hot.tolist()} -> {text_label}")
            #print("text_labels2",text_labels2)
            #print("text_labels2 type",type(text_labels2))
            


            ######




            #___________________
            text_features, logits1, logits2,logit2,text_features_aftervisualprompt,realanomalyembeddings,normal_enhanced_embeddings,all_S_n_n,all_S_n_a,normal_lengths = model(visual_feat, None, prompt_text, feat_lengths,text_labels2) 
            #print("shape text_features_aftervisualprompt",text_features_aftervisualprompt.shape) #([96, 7, 512])
            #print("text_labels2 shape",len(text_labels2)) #96
 #________________________________________________________________________________________
            

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

                    #print("abnormal text_labels2[j]",text_labels2[j]) 
                    normalized_text_feature_abr = torch.cat((normalized_text_feature_abr,realanomalyembeddings[j].unsqueeze(0)), dim=0)#/ realanomalyembeddings[j].norm(dim=-1, keepdim=True).unsqueeze(1)  
                    normalized_visual_feature_abr = visual_feat[j]# / visual_feat[j].norm(dim=-1, keepdim=True) # anomaly visual features from clip
                    current_textlabel=text_labels2[j]
                    #print("current_textlabel",current_textlabel)
                    # Check if the current label is one of the specific labels
 # Check if the current label is one of the specific labels
                    if any(lbl in exclude_labels for lbl in current_textlabel.split()) or  current_textlabel in exclude_labels:
                        # Filter out the rest of the text labels except for the current_textlabel components
                        #current_labels = current_textlabel.split()
                        otherlabels = [l for l in exclude_labels if l not in current_textlabel]
                        #print("otherlabels",otherlabels)
                        #print("otherlabels type",type(otherlabels))

                        # Encode the remaining labels
                        otherencodedtextlabels_abr = model.encode_textprompt(otherlabels)
                        #print("otherencodedtextlabels_abr shape",otherencodedtextlabels_abr.shape)
                        #otherencodedtextlabels_abr_=torch.cat((otherencodedtextlabels_abr_, otherencodedtextlabels_abr.unsqueeze(0)), dim=0)
                                    # Pad the encoded text labels with zeros if necessary
                        if otherencodedtextlabels_abr.shape[0] < max_exclude_labels - 1:
                            padding = torch.zeros((max_exclude_labels - 1 - otherencodedtextlabels_abr.shape[0], 512)).to(device)
                            otherencodedtextlabels_abr = torch.cat((otherencodedtextlabels_abr, padding), dim=0)
                        

                        
                        otherencodedtextlabels_abr_ = torch.cat((otherencodedtextlabels_abr_, otherencodedtextlabels_abr.unsqueeze(0)), dim=0)

                        #print("otherencodedtextlabels_abr_ shape",otherencodedtextlabels_abr_.shape)
                    #print("visual_feat[j]",visual_feat[j])
                    #print("normalized_visual_feature_abr shape",normalized_visual_feature_abr.shape)
                    #text_feature_abr_real = torch.cat((text_feature_abr_real, normalized_text_feature_abr.unsqueeze(0)), dim=0)
                    text_feature_abr_real=normalized_text_feature_abr
                    #print("text_feature_abr_real ",text_feature_abr_real.shape)
                    visual_features_abr = torch.cat((visual_features_abr, normalized_visual_feature_abr.unsqueeze(0)), dim=0)
                    abnormal_indices.append(j)
            #print("normal_indices",normal_indices)
            #print("abnormal_indices",abnormal_indices)

            #normal_enhanced_embeddings=normal_enhanced_embeddings/normal_enhanced_embeddings.norm(dim=-1, keepdim=True)

            #print("normal_enhanced_embeddings shape",normal_enhanced_embeddings.shape) #([ 512])
            #print("text_feature_abr_real shape",text_feature_abr_real.shape) #text_feature_abr_real shape torch.Size([64, 512, 1])
            otherencodedtextlabels_abr_=otherencodedtextlabels_abr_.permute(0, 2, 1)
            #normal_enhanced_embeddings=normal_enhanced_embeddings.unsqueeze(-1)
            text_feature_abr_real=text_feature_abr_real.unsqueeze(-1)
            #print("text_feature_abr_real shape",text_feature_abr_real.shape) #text_feature_abr_real shape torch.Size([64, 512, 1])

            #print("normal_enhanced_embeddings shape",normal_enhanced_embeddings.shape) #([ 512])

            #print("otherencodedtextlabels_abr_ shape",otherencodedtextlabels_abr_.shape)

            #print("visual_features_abr shape",visual_features_abr.shape)#visual_features_abr shape torch.Size([64, 256, 512])


           
            S_a_a=visual_features_abr @otherencodedtextlabels_abr_.type(visual_features_abr.dtype)

            #print("S_a_a shape",S_a_a.shape)
            S_a_n=normalityTextual_anomalyVisualfeatures_alignment=visual_features_abr @ normal_enhanced_embeddings.type(visual_features_abr.dtype) #/ 0.07# Alignment map between text features and visual features
            S_a_a_org=anomalyTextual_anomalyVisualfeatures_alignment=visual_features_abr @ text_feature_abr_real.type(visual_features_abr.dtype) #/ 0.07# Alignment map between text features and visual features
            #print("normalityTextual_anomalyVisualfeatures_alignment",normalityTextual_anomalyVisualfeatures_alignment)
            #print("anomalyTextual_anomalyVisualfeatures_alignment",anomalyTextual_anomalyVisualfeatures_alignment)
            
            #print("normalityTextual_anomalyVisualfeatures_alignment norm shape before",normalityTextual_anomalyVisualfeatures_alignment)#normalityTextual_anomalyVisualfeatures_alignment shape torch.Sizee([64, 256, 1]) ([X : abnormal input size, 256])
            #print("anomalyTextual_anomalyVisualfeatures_alignment norm.shape before",anomalyTextual_anomalyVisualfeatures_alignment)#anomalyTextual_anomalyVisualfeatures_alignment.shape torch.Size([64, 256, 1]) --> ([X : abnormal input size, 256, 1])
            
            #print("normalityTextual_anomalyVisualfeatures_alignment shape",normalityTextual_anomalyVisualfeatures_alignment.shape)#normalityTextual_anomalyVisualfeatures_alignment shape torch.Sizee([64, 256, 1])
            #print("anomalyTextual_anomalyVisualfeatures_alignment.shape",anomalyTextual_anomalyVisualfeatures_alignment.shape)#anomalyTextual_anomalyVisualfeatures_alignment.shape torch.Size([64, 256, 1])
            # Normalize the alignment maps 
            normalityTextual_anomalyVisualfeatures_alignment = normalityTextual_anomalyVisualfeatures_alignment / (normalityTextual_anomalyVisualfeatures_alignment.norm(dim=-1, keepdim=True)+ 1e-8)
            anomalyTextual_anomalyVisualfeatures_alignment = anomalyTextual_anomalyVisualfeatures_alignment / (anomalyTextual_anomalyVisualfeatures_alignment.norm(dim=-1, keepdim=True) +  1e-8)
            
            #print("normalityTextual_anomalyVisualfeatures_alignment norm shape",normalityTextual_anomalyVisualfeatures_alignment)#normalityTextual_anomalyVisualfeatures_alignment shape torch.Sizee([64, 256, 1]) ([X : abnormal input size, 256])
            
            # Define the guidance weight α
            alpha = 0.2  # You can choose a different value if needed
            #print("normalityTextual_anomalyVisualfeatures_alignment.squeeze(1)",normalityTextual_anomalyVisualfeatures_alignment.unsqueeze(-1).shape)
            # Compute phi as per the formula in the TPVAD
            psi_i = alpha * anomalyTextual_anomalyVisualfeatures_alignment+ (1 - alpha) * (1 - normalityTextual_anomalyVisualfeatures_alignment.unsqueeze(-1) )

            # Normalize phi 
            #psi_i = psi_i / (psi_i.norm(dim=-1, keepdim=True)+0.07)


            # Define a threshold 
            theta = 0.81 # You can choose a different value if needed

            # Obtain pseudo-labels
            gamma_a_i_j = (psi_i >= theta).float()

            #print("psi_i", psi_i)
            #print("gamma_a_i_j",gamma_a_i_j)
            #print("psi_i shape", psi_i.shape)#([X, 256, 1])

            #print("gamma_a_i_j shape", gamma_a_i_j.shape) #([X, 256, 1])
            
            # Initialize a new tensor for the result
            zeros_tensor = torch.zeros((len(text_labels2), gamma_a_i_j.shape[1], gamma_a_i_j.shape[2]), device=gamma_a_i_j.device)

            # Assign gamma_a_i_j values to the abnormal indices
            for n,idx in enumerate(abnormal_indices):
                zeros_tensor[idx] = gamma_a_i_j[n]
                #print("zeros_tensor[idx] ",idx,zeros_tensor[idx])
                #print("gamma_a_i_j[i]",n, gamma_a_i_j[n])

            #print("Result tensor:", zeros_tensor)
            gamma_a_i_j_prepended=zeros_tensor
            # Create a tensor of zeros with the specified length
            #zeros_tensor = torch.zeros((num_zeros, gamma_a_i_j.shape[1], gamma_a_i_j.shape[2]), device=gamma_a_i_j.device)

            # Concatenate the zeros tensor with the original tensor
            #gamma_a_i_j_prepended = torch.cat((zeros_tensor, gamma_a_i_j), dim=0)

            # Print the shapes to verify
            #print("gamma_a_i_j shape:", gamma_a_i_j.shape)
            #print("zeros_tensor shape:", zeros_tensor.shape)
            #print("gamma_a_i_j_prepended shape:", gamma_a_i_j_prepended.shape) #torch.Size([128, 256, 1])
            #print("gamma_a_i_j_prepended[0]", gamma_a_i_j_prepended[0])
            #print("gamma_a_i_j_prepended[64]", gamma_a_i_j_prepended[64])
            #for i in range(gamma_a_i_j_prepended):
                #if torch.any(gamma_a_i_j_prepended[i] == 0):
                    #print("gamma_a_i_j contains zero:",i, gamma_a_i_j_prepended[i])
                #else:
                    #print("gamma_a_i_j:", gamma_a_i_j_prepended[i])
                #print("logits[]:", logits1[i])
            pseudo_labels=gamma_a_i_j_prepended
                        # __________________________________REMOVING LABELING FOR THE FEATURES ABOVE THE feat_lengths
            #print("paeudo labels shape before ",pseudo_labels.shape) #[X, 256, 1]
            #print("logits1.shape",logits1.shape) #[X, 256, 1]
            Logits1 = torch.zeros(0).to(device) #
            Pseudo_labels = torch.zeros(0).to(device)
            lengths=feat_lengths
            Ph_i_lenghts =torch.zeros(0).to(device)
            normal_label_len=0
            for k in range (len(text_labels2)):
                if text_labels2[k]=='normal':
                    normal_label_len+=1
            # print(normal_label_len)
            # print(len(text_labels2))
            # print(psi_i.shape)
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
            c=0
            expectations=[]
            # Ph_i_lengths_list = Ph_i_lenghts.squeeze().tolist()
            # print(len(Ph_i_lengths_list)) 
            # Loop through each anomaly length
            for length in abnormal_lengths:
                end_idx = start_idx + length  # End index for slicing

 

                mean_value = torch.mean(Ph_i_lenghts[start_idx:end_idx])
                expectations.append(mean_value)
                values = Ph_i_lenghts[start_idx:end_idx].detach().cpu().numpy()
 
                # Set the threshold as the 90th percentile
                percentile_threshold = np.percentile(values, 80)
                c+=1
                # print(values,c)

                # print("90th percentile threshold:", percentile_threshold,c)

                # Classify frames based on this percentile threshold
                anomaly_labels = torch.tensor(values >= percentile_threshold).float()
                # print(anomaly_labels.shape,"new labels lengths ")


                all_anomaly_labels.extend(anomaly_labels)

                # Update the start index for the next slice
                start_idx = end_idx

            PL_tensor2 = torch.zeros(0).to(device)  # Initialize an empty tensor on the specified device
            start2 = 0  # To keep track of the starting index for anomaly labels

            for x, lbl in enumerate(text_labels2):
                feat_length = feat_lengths[x]  # Length of current feature set
                # print(feat_length)
                
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
            # print(gamma.shape)
                        # print(gamma)

            zeros_tensor3 = torch.zeros((len(text_labels2), gamma.shape[1], gamma.shape[2]), device=gamma.device)

            # Assign gamma_a_i_j values to the abnormal indices
            for ns,idxs in enumerate(abnormal_indices):
                zeros_tensor3[idxs] = gamma[ns]

            Pseudo_labels3 = torch.zeros(0).to(device)
            
            for m in range(zeros_tensor3.shape[0]):
                tmp_33=zeros_tensor3[m, 0:lengths[m]]
                Pseudo_labels3=torch.cat([Pseudo_labels3,tmp_33 ],dim=0)        

            # print(feat_lengths)
            # print(PL_tensor2[-346:])
            #__________________________
            # loss_4_BCE = F.binary_cross_entropy_with_logits( Logits1,Pseudo_labels)
            #loss_4_BCE = F.binary_cross_entropy_with_logits( logits1,pseudo_labels)
            # Logits1 = Logits1.squeeze(dim=1)
            # loss_4_BCE = F.binary_cross_entropy_with_logits( Logits1,PL_tensor2) # with the 80 thrshold 
            loss_4_BCE = F.binary_cross_entropy_with_logits( Logits1,Pseudo_labels3) # with the mean threshold 

            loss_total4 += loss_4_BCE.item()

            #print("BCE_new_loss",loss_total4)
            #print("loss_total1",loss_total1)
            #print("loss_total2",loss_total2)
            #print("loss_total3",loss3)
#_______________________
            L_rank_n, L_rank_a = compute_ranking_loss(all_S_n_n, all_S_n_a, S_a_n, S_a_a_org, S_a_a,normal_lengths,abnormal_lengths)
            #print("L_rank_n",L_rank_n)
            #print("L_rank_a",L_rank_a)
            loss_rank_n+=L_rank_n.item()
            loss_rank_a+=L_rank_a.item()













            #_________________________
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)

            #loss2,labels_cat = CLASM(logit2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6

            #________
                        #________________Categorical pseudo labeling__________________ 
            #print("text_labels2 shape",len(text_labels2))
            #print("pseudo_labels shape",len(pseudo_labels))
            #print("pseudo_labels shape",pseudo_labels.shape) #[128, 256, 1]
            #print("text_labels2 ",text_labels2)
            #print("pseudo_labels ",pseudo_labels[0][0]) 
            #print("logits2.shape",logits2.shape) #([128, 256, 14])
            #print("logits2[0,7]",logits2[0,7])
            # print("text_labels[65,0]",text_labels[0]) #torch.Size([128, 14])
            
            # Initialize Categorical_pseudo_labels with the same shape as logits2
            Categorical_pseudo_labels = torch.zeros(logits2.shape, dtype=torch.float32, device='cuda')
            PL_tensor2 = PL_tensor2.unsqueeze(dim=1)  # Add an extra dimension to make it (X, 1)
            # print(PL_tensor2[0])
            # print(PL_tensor2.shape)
            # print(Pseudo_labels3.shape)
            # print("pseudo_labels[l][m]",pseudo_labels.shape)

            #print("Categorical_pseudo_labels shape",Categorical_pseudo_labels.shape)
            counting=0
            counting2=0
            # for l in range(text_labels.shape[0]):#128
            #     for m in range(PL_tensor2.shape[0]):#256
            #     #for m in range( pseudo_labels.shape[1]):#256     
            #         #print("pseudo_labels[l][m]",pseudo_labels[l][m].item())
            #         if PL_tensor2[m]  == 0:
            #         #if pseudo_labels[l][m].item()  == 0:

            #             # print("1",PL_tensor2[l][m])
            #             #print("Categorical_pseudo_labels[l,m] shape" , Categorical_pseudo_labels[l,m,:].shape)
            #             #print("text_labels[0]" , text_labels[l].shape)
            #             Categorical_pseudo_labels[l,m] = torch.tensor([1., 0., 0., 0., 0., 0., 0.])
                        
            #             #print("2",Categorical_pseudo_labels[l,m])
            #             #print("normal Categorical_pseudo_labels[l,m]",Categorical_pseudo_labels[l,m])
            #             counting+=1
            #             #print("normal Categorical_pseudo_labels[l,m] count",counting)
            #         else:
            #             Categorical_pseudo_labels[l,m] = text_labels[l]
            #             #print("3",Categorical_pseudo_labels[l,m])
            #             #print("original text label ", text_labels2[l])
            #             #print("abnormal Categorical_pseudo_labels[l,m]",Categorical_pseudo_labels[l,m])
            #             counting2+=1
                        #print("abnotmal Categorical_pseudo_labels[l,m] count",counting2)
                    #if l ==65:
                        #print("Categorical_pseudo_labels[l,m] for 65",Categorical_pseudo_labels[l,m])
                #print("done",l,"of 128")
            # Verify the shape of the result
            #print(Categorical_pseudo_labels.shape)  #([128, 256, 14])
            #print("Categorical_pseudo_labels[65]",Categorical_pseudo_labels[65])
            #print("counting1",counting)
            #print("counting2",counting2)
            # Now calculate the cross-entropy loss
            # print("PL_tensor2",PL_tensor2[936:997])
            # print(sum(feat_lengths)==PL_tensor2.shape[0])
# Define the tensor to assign when PL_tensor2[pp] == 0 to avoid recreating it each time
            default_tensor = torch.tensor([1., 0., 0., 0., 0., 0., 0.])

            # Initialize an offset to track the position in PL_tensor2
            offset = 0

            # Loop over each video's frame length
            for l in range(len(feat_lengths)):  # Iterate over each video label
                current_length = feat_lengths[l]  # Number of frames for the current video
                
                # Iterate over each frame within the current video's feature length
                for pp in range(current_length):
                    # Check the value in PL_tensor2 / Pseudo_labels3 at the current offset position
                    if Pseudo_labels3[offset + pp] == 0:
                        Categorical_pseudo_labels[l, pp] = default_tensor
                         
                    else:
                        Categorical_pseudo_labels[l, pp] = text_labels[l]
                        # print("Yesss")
                offset += current_length
            # print(text_labels2)

            # print(feat_lengths)            


            # for h in range ( 257):
            #     print("Categorical_pseudo_labels", Categorical_pseudo_labels[0,h])



            
            Logits2 = torch.zeros(0).to(device)
            Pseudo_labels_cat = torch.zeros(0).to(device)
            for p in range(logits2.shape[0]):
                tmp_=logits2[p, 0:lengths[p]]
                Logits2=torch.cat([Logits2,tmp_] ,dim=0)
                tmp_2= Categorical_pseudo_labels[p, 0:lengths[p]]
                Pseudo_labels_cat=torch.cat([Pseudo_labels_cat,tmp_2 ],dim=0)
            #print("Categorical_pseudo_labels shape before ",Categorical_pseudo_labels.shape)

            #print("Categorical_pseudo_labels shape after ",Pseudo_labels_cat.shape)
            #print("logits2 shape  after ",Logits2.shape)

            # for i in range(0, 256):
            #     print("Categorical_pseudo_labels", Categorical_pseudo_labels[i])

            # Reshape logits and labels for cross-entropy loss
            Logits2 = Logits2.view(-1, logits2.shape[2])  # Reshape to (X*256, 14)
            Pseudo_labels_cat = Pseudo_labels_cat.view(-1,Categorical_pseudo_labels.shape[2])     #  Reshape to (X*256, 14)
            #print("Pseudo_labels_cat shape after",Pseudo_labels_cat.shape)
            #print("logits2 shape sfter",Logits2.shape)

            # for u in range(0, 256):
            #     print("Categorical_pseudo_labels", Pseudo_labels_cat[u])

#________________
            # labels_cat=labels_cat.unsqueeze(1).repeat(1, 256, 1) #labels_cat : abnormal cat labels
            # labels_cat_=torch.zeros(0).to(device)
            # Logits2_ = torch.zeros(0).to(device)
            # for p in range(logit2.shape[0]):
            #     tmp_=logit2[p, 0:abnormal_lengths[p]]
            #     tmp2_=labels_cat[p, 0:abnormal_lengths[p]]
            #     Logits2_=torch.cat([Logits2_,tmp_] ,dim=0)
            #     labels_cat_=torch.cat([labels_cat_,tmp2_] ,dim=0)


           # loss5_CE= F.binary_cross_entropy_with_logits(Logits2_, labels_cat_)

            loss5_CE= F.binary_cross_entropy_with_logits(Logits2, Pseudo_labels_cat)




            #loss5_CE=F.cross_entropy(Logits2, Pseudo_labels_cat)
            #loss5_CE = criterion(logits2, Categorical_pseudo_labels_indices)
            #loss5_CE = CLASM_CCE(logits2, Categorical_pseudo_labels)
            #loss5_CE=F.cross_entropy(logits2, Categorical_pseudo_labels_indices)


            # Print the loss
            #print("Cross-entropy loss:", loss5_CE.item())
            loss_total5 += loss5_CE.item()

            #loss = loss1 + loss2 + loss3 * 1e-4
            #loss = loss_4_BCE + loss2 + loss3* 1e-4+L_rank_n+L_rank_a
            loss=loss_4_BCE + loss5_CE + loss3* 1e-4+L_rank_n+L_rank_a
            #loss=loss_4_BCE + loss6_mil + loss3* 1e-4

            #print("total loss",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * train_loader.batch_size
            #print("step",step)
            if step % 4800 == 0 and step != 0:
                #print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                                #putting my new loss instead of loss 1 
                #print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                #print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total4 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item(),'| loss_rank_n: ', loss_rank_n / (i+1), '| loss_rank_a: ', loss_rank_a / (i+1))
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total4 / (i+1), '| loss2: ', loss_total5 / (i+1), '| loss3: ', loss3.item(),'| loss_rank_n: ', loss_rank_n / (i+1), '| loss_rank_a: ', loss_rank_a / (i+1))
                #print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total4 / (i+1), '| loss2: ', loss_total6 / (i+1), '| loss3: ', loss3.item())

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


    