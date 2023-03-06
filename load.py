import torch
from collections import OrderedDict

new_dict = OrderedDict()

# load_net_1 = torch.load('C:/Users/user/Desktop/HAZE/experiment/eth_training_1/model/model_best.pt')
# load_net_1 = torch.load('F:\DynaVSR-master\pretrained_models\BasicVSR/vimeo_0404_baseline.pth')
load_net_1 = torch.load('F:/DynaVSR-master/pretrained_models/NewBasic/basic2002_x_3_1_G.pth')
load_net_2 = torch.load('F:/DynaVSR-master/pretrained_models/NewBasic/basic0220_y_1_3_G.pth')
# basic0220_y_3_1_G
# load_net_2 = torch.load('C:/Users/user/Desktop/HAZE/experiment/eth_training_1/model/SR_model_best.pt')

# load_net_2 = torch.load('F:/KAWM/pretrained\HAN/HAN0240_new.pth')
# load_net_2 = torch.load('F:\KAWM\experiments\RCAN_0240_new_4002\models/43000_G.pth')
# load_net_2 = torch.load('F:\KAWM\pretrained\RCAN/RCAN0240_new.pth')
# load_net_2 = torch.load('./pretrained/HAN/HAN0240.pth')

# print(load_net_1['h2a2sr.R2.body.3.conv_du.2.weight'][0])
# print('---')
# print(load_net_2['h2a2sr.R2.body.3.conv_du.2.weight'][0])
# # print(load_net_1['body.1.body.10.kawm.transformer.weight'])
for k, v in load_net_1.items():
    # print(k)
    # if 'spynet' in k:
    new_dict[k] = v
    
for k, v in load_net_2.items():
    if 'spynet' in k:
        continue
    new_dict[k] = v

for k, v in new_dict.items():
    print(k)
# # # print(new_dict['body.1.body.10.kawm.transformer.weight'])

torch.save(new_dict, 'F:/DynaVSR-master/pretrained_models/NewBasic/merged_vimeo_S2_v2.pth')
