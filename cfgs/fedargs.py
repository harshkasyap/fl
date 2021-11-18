import asyncio, inspect, os, sys
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import agg, fl, nn, poison, resnet, text_utils

argsdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class FedArgs():
    def __init__(self):
        self.name = "client-x"
        self.num_clients = 50
        self.epochs = 51
        self.local_rounds = 1
        self.client_batch_size = 32
        self.test_batch_size = 128
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.cuda = False
        self.seed = 1
        self.loop = asyncio.get_event_loop()
        self.agg_rule = agg.Rule.FedAvg
        self.dataset = "mnist" # can run for mnist, f-mnist, cifar-10, for ag_news ?
        self.one_d_len = 784
        self.labels = [label for label in range(10)] # for mnist and f-Mnist
        self.model = nn.ModelMNIST() # for mnist and f-mnist #resnet.ResNet18() for cifar - 10
        self.train_func = fl.train_model
        self.eval_func = fl.evaluate
        self.topic = "pyflx"
        self.broker_ip = '172.16.26.40:9092'
        self.schema_ip = 'http://172.16.26.40:8081'
        self.wait_to_consume = 10
        self.hdc_proj_len = 10000
        self.tb = SummaryWriter(argsdir + '/../out/runs/fl/test-run', comment="fl")
        
fedargs = FedArgs()

class TextConfig(object):
    embed_size = 300
    num_channels = 100
    kernel_size = [3,4,5]
    output_size = 4
    lr = 0.3
    batch_size = 64
    max_sen_len = 30
    dropout_keep = 0.2
    
textconfig = TextConfig()
        
# FLTrust
FLTrust = {"is": True if fedargs.agg_rule in [agg.Rule.FLTrust, agg.Rule.FLTC] else False,
           "ratio": 0.003,
           "data": None,
           "loader": None,
           "proxy": {"is": False,
                     "ratio": 0.5,
                     "data": None,
                     "loader": None}}

# No of malicious clients
mal_clients = [c for c in range(20)]

# Label Flip
label_flip_attack = {"is": False,
                     "func": poison.label_flip_next, # see other possible functions in poison python file
                     "labels": {},
                     "percent": -1}

def set_lfa_labels(flip_labels = None):
    if flip_labels is None:
        label_flip_attack["labels"] = {4: 6} if label_flip_attack["is"] and label_flip_attack["func"] is poison.label_flip else None
        label_flip_attack["labels"] = {label: fedargs.labels[(index + 1) % len(fedargs.labels)] for index, label in enumerate(fedargs.labels)} if label_flip_attack["is"] and label_flip_attack["func"] is poison.label_flip_next else label_flip_attack["labels"]
    else:
        label_flip_attack["labels"] = flip_labels

# Backdoor
backdoor_attack = {"is": False,
                   "trojan_func": poison.insert_trojan_pattern, # see other possible functions in poison python file
                   "target_label": 6,
                   "ratio": 0.006,
                   "data": None,
                   "loader": None}

# Layer replacement attack
layer_replacement_attack = {"is": False,
                            "layers": ["conv1.weight"],
                            "func": poison.layer_replacement_attack}

# Cosine attack
cosine_attack = {"is": False,
                 "args": {"poison_percent": 1,
                          "scale_dot": 5,
                          "scale_dot_factor": 1,
                          "scale_norm": 500,
                          "scale_norm_factor": 2,
                          "scale_epoch": 5},
                 "kn": poison.Knowledge.IN,
                 "func": poison.sine_attack}

# Fang attack proposed for trimmed mean
fang_attack = {"is": False,
               "kn": poison.Knowledge.PN,
               "func": poison.fang_trmean}

# LIE attack
lie_attack = {"is": False,
              "kn": poison.Knowledge.PN,
              "func": poison.lie_attack}

# SOTA attack proposed for trimmed mean, min-max, min-sum
sota_attack = {"is": False,
               "kn": poison.Knowledge.PN,
               "dev_type": "unit_vec", # see other possible functions in poison python file
               "func": poison.sota_agnostic_min_max # see other possible functions in poison python file
               }

# Sybil attack, for sending same update as base
sybil_attack = {"is": False}