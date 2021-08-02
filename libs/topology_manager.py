import copy
import os
import sys
import time
import socket
from builtins import len, range, super

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import nn, log
from libs.protobuf_producer import *
from libs.protobuf_consumer import *

#!pip install syft
import syft as sy
from syft.federated.model_serialization import State
from syft.federated.model_serialization import wrap_model_params

class TopologyManager:
    """
    Arguments:
        adj_mat: adjacency matrix from the given network
    """

    def __init__(self, adj_mat, clients, node_types):
        self.topology = []
        self.group_prefix = "group-"
        self.client_prefix = "client-"        
        self.adj_mat = adj_mat[:]
        self.clients = clients[:]
        self.num_nodes = len(self.adj_mat)
        self.rcvd_models = [{} for client in range(len(self.adj_mat))]

    # @todo This will break adj_mat concept, need to correct.
    def is_node_avail(self, node_index):
        if node_index in self.clients:
            return True
        log.error("Client {} is not available for training".format(node_index))
        return False

    def get_rcvd_models(self, node_index):
        _rcvd_models = []
        # @todo update rcvd_models as dictionary, once adj_mat concept is fixed
        if self.is_node_avail(node_index):
            _rcvd_models = self.rcvd_models[node_index]
        return _rcvd_models


    def flush_models(self):
        self.rcvd_models = [{} for _ in range(len(self.adj_mat))]

    def get_aggregators(self):
        aggr_nodes = self.node_type['aggregator']
        return aggr_nodes

    def get_trainers(self):
        trainer_nodes = self.node_type['trainer']
        return trainer_nodes

    def get_broadcaster(self):
        broadcaster_nodes = self.node_type['broadcaster']
        return broadcaster_nodes
    
    def produce_model(self, node_index, _topic, model, epoch):
        group_id = self.group_prefix + node_index
        client_id = self.client_prefix + node_index
        pb = sy.serialize(wrap_model_params(model.parameters()))
        protobuf_producer = ProtobufProducer(group_id, client_id, State.get_protobuf_schema())
        try:
            protobuf_producer.produce(topic=_topic, 
                                      _key=str(node_index), 
                                      _value=pb, 
                                      _headers={'iteration': str(epoch)})
        except Exception as se:
            log.error("Exception {} occured producing update for client {}".format(se, node_index))
            
    def consume_model(self, node_index, _topic, _model, _epoch):
        _rcvd_models = {}

        if self.is_node_avail(node_index):
            group_id = self.group_prefix + node_index
            client_id = self.client_prefix + node_index
            protobuf_consumer = ProtobufConsumer(group_id, client_id, State.get_protobuf_schema(), _topic)

            try:
                t_end = time.time() + 10
                while time.time() < t_end:
                    msg = protobuf_consumer.consume()
                    epoch = -1
                    if msg is not None:
                        if msg.headers() is not None:
                            for tup in msg.headers():
                                if tup[0] == 'iteration' and tup[1] is not None:
                                    epoch = tup[1].decode('utf-8')

                    if str(epoch) == str(_epoch):
                        model = nn.getModel(msg.value(), _model)
                        _rcvd_models[msg.key()] = copy.deepcopy(model)
            except KeyboardInterrupt:
                log.error("Exception KeyboardInterrupt occured consuming update for client {}".format(node_index))
            except Exception as se:
                log.error("Exception {} occured consuming update for client {}".format(se, node_index))
            finally:
                protobuf_consumer.close()
        
        return _rcvd_models



class CentralizedTopology(TopologyManager):
    def __init__(self, adj_mat, server, clients, node_types):
        super().__init__(adj_mat, clients, node_types)
        self.clients.append(server)

    def get_neigh_weights(self, node_index):
        neigh_weights = []
        if self.is_node_avail(node_index):
            neigh_weights = self.adj_mat[node_index]
        return neigh_weights

    def get_neigh_list(self, node_index):
        neigh_weights = self.get_neigh_weights(node_index)
        neigh_list = [index for (index, weight) in enumerate(neigh_weights) if weight]
        return neigh_list

    def broadcast(self, node_index, model):
        if self.is_node_avail(node_index):
            # @todo need to fix, how to pass adj_mat
            neigh_list = self.get_neigh_list(node_index)
            for node_dst in neigh_list:
                self.rcvd_models[node_dst][node_index] = copy.deepcopy(model)


class DistributedTopology(TopologyManager):
    def __init__(self, adj_mat, clients, node_types):
        super().__init__(adj_mat, clients, node_types)

    def get_in_neigh_weights(self, node_index):
        if node_index >= self.num_nodes:
            return []
        in_neighbor_weights = []
        for row_idx in range(len(self.adj_mat)):
            in_neighbor_weights.append(self.adj_mat[row_idx][node_index])
        return in_neighbor_weights

    def get_out_neigh_weights(self, node_index):
        if node_index >= self.num_nodes:
            return []
        return self.adj_mat[node_index]

    def get_in_neigh_list(self, node_index):
        neigh_in_list = []
        neighbor_weights = self.get_in_neigh_weights(node_index)
        for idx, neigh_w in enumerate(neighbor_weights):
            if neigh_w > 0 and node_index != idx:
                neigh_in_list.append(idx)
        return neigh_in_list

    def get_out_neigh_list(self, node_index):
        neigh_out_list = []
        neighbor_weights = self.get_out_neigh_weights(node_index)
        for idx, neigh_w in enumerate(neighbor_weights):
            if neigh_w > 0 and node_index != idx:
                neigh_out_list.append(idx)
        return neigh_out_list