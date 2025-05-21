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
    #print("labels.shape",labels.shape)
    #print("labels",labels)
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

    #print("labels_.shape",labels_.shape) # (X,14)
    #print("labels_",labels_)
    #print("lengths_",lengths)
    #print("lengths_",lengths_)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:int(lengths_[i].item())], k=int(int(lengths_[i].item()) / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    #print("instance_logits.shape",instance_logits.shape) # torch.Size([64, 14])
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


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    #print("gt length",len(gtlabels))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    #print("prompt_text",prompt_text) # all including the normal =14 classes 
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
        loss_total6=0
        loss_rank_n=0 
        loss_rank_a=0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        #print("min(len(normal_loader), len(anomaly_loader", len(normal_loader), len(anomaly_loader))
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter) # to fetch the next batch
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)
            #print("normal_features shape",normal_features.shape) #([64, 256, 512])
            #print("anomaly_features shape",anomaly_features.shape)#([64, 256, 512])
            #print("anomaly_lengths",anomaly_lengths)# a list of differnt lengths of the clip/ segment for each normal features #[185, 185, 185, 185, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,....]
            #print("normal_lengths",normal_lengths)
            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device) # uses both features from normal and abnormal 
            text_labels = list(normal_label) + list(anomaly_label)
            text_labels2 = list(normal_label) + list(anomaly_label) #real anomaly labels
            #print("text_labels2 shape",len(text_labels2)) # 128
            #print("text_labels2 shape",text_labels2) # 128
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            #print("feat_lengths",len(feat_lengths)) #=128 element [ 30,  30, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122,  24,  24,....,]
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            #print("text_labels",text_labels) # one hot encoding labels
            #print("text_labels shape",text_labels.shape) # ([128, 14])
            #print("visual_features",visual_features)

            #print("visual_features shape",visual_features.shape) #clip visual_features shape torch.Size([128, 256, 512])

            #orginal,logits1,2,all textualfeatures after the visual prompt module operation :=FFN(ADD(V,tout))+tout
            text_features, logits1, logits2,logit2,text_features_aftervisualprompt,realanomalyembeddings,normal_enhanced_embeddings,all_S_n_n,all_S_n_a,normal_lengths = model(visual_features, None, prompt_text, feat_lengths,text_labels2) 
            #loss1
            #print("realanomalyembeddings",realanomalyembeddings) #
            #print("realanomalyembeddings shape",realanomalyembeddings.shape)#([128, 512]) embeddings of the class label of the current batch after the clipembeddings" learnable prompt"
            #print("text_features_aftervisualprompt shape",text_features_aftervisualprompt.shape)#([128, 14, 512])
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            #print("logits1 shape",logits1.shape) #([128, 256, 1])
            #print("loss1",loss1)
            #loss2
            #print("logits2",logits2.shape) # including the normal #([X, 256, 14])

            loss2,labels_cat = CLASM(logit2, text_labels, feat_lengths, device)
            #print("labels_cat",labels_cat.shape)
            loss_total2 += loss2.item()
            #print("logits2[0]",logits2[0])
            #print("loss2",loss2)
            #loss3
            #This loss is a regularization term to enforce similarity between the first text feature (assumed to be a normal class) and other text features.
            loss3 = torch.zeros(1).to(device)
            #Normalize the first text feature " normal" and compute the similarity with all other text features.

            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)

            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True) # from orginal textual descriptions
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            #Average the accumulated similarity losses and scale.
            # print("loss3 before",loss3)
            loss3 = loss3 / 13 * 1e-1



             #my add #________________________________________________________________________________________
            

            # Initialize tensors to store the values
            text_feature_abr_real = torch.zeros(0).to(device)
            visual_features_abr = torch.zeros(0).to(device)
            num_zeros=0
            # Iterate through the text labels and concatenate the values to the corresponding tensors
            # Initialize an empty list to store the indices
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
                    #print("noemal text_labels2",text_labels2[j])

                else:
                    abnormal_lengths.append(feat_lengths[j])
                    #print("abnormal text_labels2[j]",text_labels2[j]) 
                    normalized_text_feature_abr = torch.cat((normalized_text_feature_abr,realanomalyembeddings[j].unsqueeze(0)), dim=0)#/ realanomalyembeddings[j].norm(dim=-1, keepdim=True).unsqueeze(1)  
                    normalized_visual_feature_abr = visual_features[j]# / visual_feat[j].norm(dim=-1, keepdim=True) # anomaly visual features from clip
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
            #print("abnormal_lengths",len(abnormal_lengths))
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

            for p in range(pseudo_labels.shape[0]):
                tmp_=logits1[p, 0:lengths[p]]
                Logits1=torch.cat([Logits1,tmp_ ],dim=0)
                tmp_2= pseudo_labels[p, 0:lengths[p]]
                Pseudo_labels=torch.cat([Pseudo_labels,tmp_2 ],dim=0)
                if text_labels2[p]!='Normal':
                    # print("lengths[p]",lengths[p])
                    tmp_3= psi_i[p-len(normal_label), 0:lengths[p]]
                    Ph_i_lenghts = torch.cat([Ph_i_lenghts, tmp_3], dim=0)
            #print("paeudo labels shape before ",pseudo_labels.shape)

            #print("paeudo labels shape after ",Pseudo_labels.shape)
            #print("logits  shape after ",Logits1.shape)

            #print("pseudo labels",Pseudo_labels)
            all_anomaly_labels=[]
            start_idx = 0  # Start index for slicing
            c=0
            # Ph_i_lengths_list = Ph_i_lenghts.squeeze().tolist()
            # print(len(Ph_i_lengths_list)) 
            # Loop through each anomaly length
            for length in anomaly_lengths:
                end_idx = start_idx + length  # End index for slicing

                # Compute the mean for 
                # the slice corresponding to this anomaly length
                # mean_value = torch.mean(Ph_i_lenghts[start_idx:end_idx])
                # mean_value=np.percentile(Ph_i_lenghts[start_idx:end_idx], 90) 
                        # Plot the distribution

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
                
                if lbl == 'Normal':
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
            # print(PL_tensor2[-250:])
            # print("Ones indices 3",torch.nonzero(PL_tensor2[-250:-70]  == 1).squeeze())
            # print("Ones indices 2",torch.nonzero(PL_tensor2[-70:-18]  == 1).squeeze())
            # print("Ones indices 1",torch.nonzero(PL_tensor2[-18:]  == 1).squeeze())
            # # anomaly_y=PL_tensor2[-18:]
            # anomaly_y = anomaly_y.cpu().numpy()  # Assuming it's a PyTorch tensor

            # # Plot setup
            # fig, ax = plt.subplots(figsize=(10, 4))
            # ax.set_xlim(0, 17)
            # ax.set_ylim(-0.2, 1.2)
            # ax.set_xlabel('Frame Index')
            # ax.set_ylabel('Anomaly Label (0 = Normal, 1 = Anomaly)')
            # ax.set_title('Anomaly Labels Representation per Frames')
            # ax.set_xticks(list(range(18)))
            # ax.grid(True)
            # ax.legend(['Anomaly Labels'])

            # line, = ax.plot([], [], 'bo-', lw=2)

            # # Initialize function for animation
            # def init():
            #     line.set_data([], [])
            #     return line,

            # # Animation function: called sequentially
            # def animate(i):
            #     x = list(range(i + 1))
            #     y = anomaly_y[:i + 1]
            #     line.set_data(x, y)
            #     return line,

            # # Creating the animation
            # ani = FuncAnimation(fig, animate, frames=len(anomaly_y), init_func=init,
            #                     blit=True, repeat=False, interval=500)  # Adjust interval for speed

            # # To save the animation as a video file
            # ani.save('anomaly_labels_animation.gif', writer='pillow', fps=2)

            # plt.show()

            # print(PL_tensor2[0:250])
            # print("lenghts of logits and PL new",Logits1.shape,PL_tensor2.shape)
            #__________________________
            # loss_4_BCE = F.binary_cross_entropy_with_logits( Logits1,Pseudo_labels)
            # Ensure Logits1 has the same shape as PL_tensor2
            Logits1 = Logits1.squeeze(dim=1)
            loss_4_BCE = F.binary_cross_entropy_with_logits( Logits1,PL_tensor2)


            #print("logits1 shape",logits1.shape)
            #print("logits1",logits1[0])

            loss_total4 += loss_4_BCE.item()
            L_rank_n, L_rank_a = compute_ranking_loss(all_S_n_n, all_S_n_a, S_a_n, S_a_a_org, S_a_a,normal_lengths,abnormal_lengths) #torch.Size([32, 256]) torch.Size([32, 256, 13]) torch.Size([32, 256]) torch.Size([32, 256, 1]) torch.Size([32, 256, 12])
            #print("L_rank_n",L_rank_n)
            #print("L_rank_a",L_rank_a)
            loss_rank_n+=L_rank_n.item()
            loss_rank_a+=L_rank_a.item()








            #print("BCE_new_loss",loss_4_BCE)
            #print("loss_total1",loss_total1)
            #print("loss_total2",loss_total2)
            #print("loss_total3",loss3)
            
            
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
            #print("Categorical_pseudo_labels shape",Categorical_pseudo_labels.shape)
            counting=0
            counting2=0
            PL_tensor2 = PL_tensor2.unsqueeze(dim=1)  # Add an extra dimension to make it (X, 1)

            for l in range(text_labels.shape[0]):#128
                # for m in range(pseudo_labels.shape[1]):#256
                for m in range(PL_tensor2.shape[1]):#256

                    #print("pseudo_labels[l][m]",pseudo_labels[l][m].item())
                    if PL_tensor2[l][m].item() == 0:
                        #print("Categorical_pseudo_labels[l,m] shape" , Categorical_pseudo_labels[l,m,:].shape)
                        #print("text_labels[0]" , text_labels[l].shape)
                        Categorical_pseudo_labels[l,m] =  torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0.])
                        #print("normal Categorical_pseudo_labels[l,m]",Categorical_pseudo_labels[l,m])
                        counting+=1
                        
                        #print("normal Categorical_pseudo_labels[l,m] count",counting)
                    else:
                        Categorical_pseudo_labels[l,m] = text_labels[l]
                        #print("abnormal Categorical_pseudo_labels[l,m]",Categorical_pseudo_labels[l,m])
                        counting2+=1
                        #print("abnotmal Categorical_pseudo_labels[l,m] count",counting2)
                    #if l ==65:
                        #print("Categorical_pseudo_labels[l,m] for 65",Categorical_pseudo_labels[l,m])
                #print("done",l,"of 128")
            # Verify the shape of the result
            #print(Categorical_pseudo_labels.shape)  #([64, 256, 14])
            #print("Categorical_pseudo_labels[65]",Categorical_pseudo_labels[65])
            #print("counting1",counting)
            #print("counting2",counting2)
            # Now calculate the cross-entropy loss


            # converting the one hot encoding categorical  pseudo labels  to class indices:
            #Categorical_pseudo_labels_indices = torch.argmax(Categorical_pseudo_labels, dim=1)
            #criterion = torch.nn.CrossEntropyLoss()
            # Compute the loss
            #loss5_CE = criterion(logits2, Categorical_pseudo_labels_indices)
            #loss5_CE = CLASM_CCE(logits2, Categorical_pseudo_labels)
            #print("logits2 shape",logits2.shape) #[64, 256, 14])
            #print("Categorical_pseudo_labels shape before ",Categorical_pseudo_labels.shape)
            #print(len(lengths))
            #print(lengths)
            #print("----------")
            #print(abnormal_lengths)
            Logits2 = torch.zeros(0).to(device)
            Pseudo_labels_cat = torch.zeros(0).to(device)
            for p in range(Categorical_pseudo_labels.shape[0]):
                tmp_=logits2[p, 0:lengths[p]]
                Logits2=torch.cat([Logits2,tmp_] ,dim=0)
                tmp_2= Categorical_pseudo_labels[p, 0:lengths[p]]
                Pseudo_labels_cat=torch.cat([Pseudo_labels_cat,tmp_2 ],dim=0)
            #print("Categorical_pseudo_labels shape before ",Categorical_pseudo_labels.shape)

            #print("Categorical_pseudo_labels shape after ",Pseudo_labels_cat.shape)
            #print("logits2 shape  before ",Logits2.shape)
            #print("Pseudo_labels_cat shape before",Pseudo_labels_cat.shape)
            #print("Pseudo_labels_cat shape before",Pseudo_labels_cat[15])

            # Convert one-hot encoded labels to class indices
            #Pseudo_labels_cat = torch.argmax(Pseudo_labels_cat, dim=-1)
            #print("Pseudo_labels_cat shape after",Pseudo_labels_cat[15])

            # Reshape logits and labels for cross-entropy loss
            Logits2 = Logits2.view(-1, logits2.shape[2])  # Reshape to (X*256, 14)
            Pseudo_labels_cat = Pseudo_labels_cat.view(-1,Categorical_pseudo_labels.shape[2])     #  Reshape to (X*256, 14)
            #print("Pseudo_labels_cat shape after",Pseudo_labels_cat.shape)
            #print("logits2 shape after",Logits2.shape)

            #____________________ 
            # Pseudo_labels_cat_=torch.zeros(0).to(device)
            # #Logits2_=torch.zeros(0).to(device)
            # for j in range(Pseudo_labels_cat.shape[0]):
            #     if Pseudo_labels_cat[j,0]==1:
                    
            #         continue
            #     else:
            #         cc+=1
            #         Pseudo_labels_cat_=torch.cat([Pseudo_labels_cat_,Pseudo_labels_cat[j].unsqueeze(0)],dim=0)
            #         #Logits2_=torch.cat([Logits2_,Logits2[j].unsqueeze(0)],dim=0)
                    
            # print(cc)
            # Pseudo_labels_cat_ = Pseudo_labels_cat_[:, 1:]
            # #Logits2_=Logits2_[:, 1:]

            #print("Pseudo_labels_cat_.shape",Pseudo_labels_cat_.shape) # (X,13)
            #print("Pseudo_labels_cat_",Pseudo_labels_cat_)
            #print("logit2 shape",logit2.shape)
            #print("cat_anomaly_labels",len(cat_anomaly_labels))
            #print("Logits2_",Logits2_)
            #for i,j in enumerate(text_labels)
            labels_cat=labels_cat.unsqueeze(1).repeat(1, 256, 1)
            labels_cat_=torch.zeros(0).to(device)
            Logits2_ = torch.zeros(0).to(device)
            for p in range(logit2.shape[0]):
                tmp_=logit2[p, 0:abnormal_lengths[p]]
                tmp2_=labels_cat[p, 0:abnormal_lengths[p]]
                Logits2_=torch.cat([Logits2_,tmp_] ,dim=0)
                labels_cat_=torch.cat([labels_cat_,tmp2_] ,dim=0)


#___________________
            loss5_CE= F.binary_cross_entropy_with_logits(Logits2_, labels_cat_)

#            loss5_CE= F.binary_cross_entropy_with_logits(Logits2, Pseudo_labels_cat)

            #loss5_CE=F.cross_entropy(Logits2, Pseudo_labels_cat)


            # Print the loss
            #print("Cross-entropy loss:", loss5_CE.item())
            loss_total5 += loss5_CE.item()

            #loss6_mil = CLASM(logits2, Categorical_pseudo_labels, feat_lengths, device)
            #loss_total6 += loss6_mil.item()
            #print("loss6_mil",loss6_mil)
              
            #____________________________________
             #loss = loss1 + loss2 + loss3
            loss = loss_4_BCE + loss2 + loss3+L_rank_n+L_rank_a
            #loss=loss_4_BCE + loss5_CE + loss3+L_rank_n+L_rank_a
            #loss=loss_4_BCE + loss6_mil + loss3

            #print("loss",loss_4_BCE.item() , loss5_CE.item()  , loss3.item(),L_rank_n.item(),L_rank_a.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(step)
            #print(len(normal_loader))
            #print(normal_loader.batch_size * 2)#128
            step += i * normal_loader.batch_size * 2
            #print("step",step)
            if step % 1280 == 0 and step != 0:
                #putting my new loss instead of loss 1 
                #print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total4 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item(),'| loss_rank_n: ', loss_rank_n / (i+1), '| loss_rank_a: ', loss_rank_a / (i+1))
                #print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total4 / (i+1), '| loss2: ', loss_total5 / (i+1), '| loss3: ', loss3.item(),'| loss_rank_n: ', loss_rank_n / (i+1), '| loss_rank_a: ', loss_rank_a / (i+1))
                #print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total4 / (i+1), '| loss2: ', loss_total6 / (i+1), '| loss3: ', loss3.item())

                AUC, AP,averageMAP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                #print("gtlabels",gtlabels)
                #print("gtlabels shape for testing",gtlabels.shape)
                AP = AUC
                #AP = averageMAP

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
    #torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)



    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    #print("normal_loader.dataset.shape",len(normal_loader.dataset)) # 8000

   
    
    
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)