def euclidean_distance(model1,model2):
  d = 0
  for param1, param2 in zip(model1.parameters(),model2.parameters()):
    if len(list(param1.shape)) != 1 and len(list(param2.shape)) != 1:
      temp = torch.cdist(param1, param2, p=2)
      d += torch.norm(temp)
  print(d)