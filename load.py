import torch
from collections import OrderedDict

new_dict = OrderedDict()

# load_net_1 = torch.load('C:/Users/user/Desktop/HAZE/experiment/eth_training_1/model/model_best.pt')
load_net_1 = torch.load('F:\DynaVSR-master\pretrained_models\BasicVSR/vimeo_0404_baseline.pth')
# load_net_2 = torch.load('C:/Users/user/Desktop/HAZE/experiment/eth_training_1/model/SR_model_best.pt')

# load_net_2 = torch.load('F:\KAWM\pretrained\HAN/HAN0240_new.pth')
# load_net_2 = torch.load('F:\KAWM\experiments\RCAN_0240_new_4002\models/43000_G.pth')
# load_net_2 = torch.load('F:\KAWM\pretrained\RCAN/RCAN0240_new.pth')
# load_net_2 = torch.load('./pretrained/HAN/HAN0240.pth')

# print(load_net_1['h2a2sr.R2.body.3.conv_du.2.weight'][0])
# print('---')
# print(load_net_2['h2a2sr.R2.body.3.conv_du.2.weight'][0])
# # print(load_net_1['body.1.body.10.kawm.transformer.weight'])
for k, v in load_net_1['state_dict'].items():
    print(k)
    new_dict[k] = v
# for k, v in load_net_2.items():
#     new_dict[k] = v

# # # print(new_dict['body.1.body.10.kawm.transformer.weight'])

# torch.save(new_dict, './pretrained/HAN/merged_HAN_new_1219.pth')
