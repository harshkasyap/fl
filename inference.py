import argparse

from libs import federated as fl, log
from process import preprocess as P
from mlp import Model_MNIST as nn_mnist

ap = argparse.ArgumentParser(description="Running Data Poisoning Attack on Federated Learning")
ap.add_argument("--savelogs", required = False, help = "Save Logs To File (info | debug)")
args = vars(ap.parse_args())
log.init(args["savelogs"], "./logs/inference.log")

class FedArgs():
  def __init__(self):
    self.num_clients = 50
    self.num_rounds = 2
    self.epochs = 2
    self.batch_size = 32
    self.learning_rate = 1e-4
    self.weight_decay = 1e-5
    self.tb = SummaryWriter(comment="Mnist Federated training")

log.logger.info("Test")
log.logger.debug("Dest")