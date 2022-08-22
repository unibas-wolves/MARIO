import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
a = torch.Tensor(5,3)
a = a.cuda()
