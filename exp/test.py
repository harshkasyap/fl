import torch

y = torch.tensor([ 8.5691,  8.5687,  8.5942,  8.5702,  8.5619,  8.5599,  8.5597, 22.7831,
         8.5673,  8.6659])

print(y)

if torch.gt(y, 25).sum() > 0:
  print("hello")