import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_MNIST(nn.Module):
  def __init__(self):
    super(Model_MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)


  def forward(self,x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output

def auditAttack(target, pred, actual_prediction, target_prediction, attack_dict):
  for i in range(len(target)):
    if target[i] == actual_prediction:
      attack_dict["instances"] += 1
    if target[i] != pred[i]:  
      if target[i] == actual_prediction and pred[i] == target_prediction:
        attack_dict["attack_success_count"] += 1
      else:
        attack_dict["misclassifications"] += 1
  
  return attack_dict

def test(model, test_loader, actual_prediction = None, target_prediction = None):
  model.eval()
  test_output = {
    "test_loss": 0,
    "correct": 0,
    "accuracy": 0,
    "attack": {
      "instances": 0,
      "misclassifications": 0,
      "attack_success_count": 0,
      "misclassification_rate": 0,
      "attack_success_rate": 0
    }
  }

  with torch.no_grad():
    for data, target in test_loader:
      output = model(data)
      test_output["test_loss"] += F.nll_loss(output, target, reduction='sum').item()
      pred = output.argmax(dim=1, keepdim=True)
      if actual_prediction != None and target_prediction != None:
        test_output["attack"] = auditAttack(target, pred, actual_prediction, target_prediction, test_output["attack"])
      test_output["correct"] += pred.eq(target.view_as(pred)).sum().item()

  test_output["test_loss"] /= len(test_loader.dataset)
  test_output["accuracy"] = (test_output["correct"] / len(test_loader.dataset)) * 100

  if actual_prediction != None and target_prediction != None:
    test_output["attack"]["attack_success_rate"] = (test_output["attack"]["attack_success_count"]/test_output["attack"]["instances"]) * 100
    test_output["attack"]["misclassification_rate"] = test_output["attack"]["misclassifications"]/test_output["attack"]["instances"]

  return test_output

def train(model, train_loader, optimizer, epochs):
  train_loss = {}
  model.train()
  for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = model(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()

    train_loss["epoch " + str(epoch + 1)] = loss.item()
  return train_loss