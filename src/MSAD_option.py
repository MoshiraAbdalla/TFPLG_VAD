import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)
parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=8, type=int) #ucf=2 #xd=8
parser.add_argument('--visual-layers', default=1, type=int) #ucf=2 #xd=1
parser.add_argument('--attn-window', default=8, type=int) #ucf=4 #xd=8
parser.add_argument('--prompt-prefix', default=10, type=int) #ucf=5 #xd=10
parser.add_argument('--prompt-postfix', default=5, type=int) #ucf=15 #xd=5
parser.add_argument('--classes-num', default=12, type=int)
parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/model_xd_mean.pth') # using xd_violence saved model
# parser.add_argument('--model-path', default='model/model_ucf_80.pth') # using UCF_Crime saved model
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--batch-size', default=96, type=int)
parser.add_argument('--test-list', default='list/MSAD_CLIP_test.csv')
parser.add_argument('--gt-segment-path', default='list/gtsegment_filtered.npy')
parser.add_argument('--gt-label-path', default='list/filtered_gt_labels.npy')
parser.add_argument('--lr', default=1e-5)#1e-5
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10])

