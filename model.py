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

class Model_Shap(nn.Module):
    def __init__(self):
        super(Model_Shap, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 100)
    
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

def test(model, test_loader, actual_prediction, target_prediction):
    model.eval()
    test_loss = 0
    correct = 0
    attack_success_count = 0
    instances = 0
    misclassifications = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(len(target)):
              if target[i] == actual_prediction:
                instances += 1
              if target[i] != pred[i]:  
                if target[i] == actual_prediction and pred[i] == target_prediction:
                  attack_success_count += 1
                else:
                  misclassifications += 1
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    attack_success_rate = attack_success_count/instances
    attack_success_rate *= 100
    misclassification_rate = misclassifications/instances

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100* acc ))
    print('Test Samples with target label {} : {}'.format(actual_prediction,instances))
    print('Test Samples predicted as  {} : {}'.format(target_prediction,attack_success_count))
    print('Test Samples with target label {} misclassified : {}'.format(actual_prediction,misclassifications))
    print("Attack success rate",attack_success_rate)
    print("misclassification_rate", misclassification_rate)
    return test_loss, acc , attack_success_rate, misclassification_rate