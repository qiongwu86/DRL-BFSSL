import os
import sys
import copy
import time
import atexit
import math
import tensorflow as tf
from .client import Client
from modules.log_manager import LogManager
from modules.data_manager import DataManager
from modules.model_manager import ModelManager
from modules.train_manager import TrainManager

from MoCo_singleGPU_copy import *


global_updates = []
class Server:

    def __init__(self, opt):
        self.opt = opt
        self.clients = {}
        self.threads = []
        self.updates = []
        self.task_names = []

        self.curr_round = -1

        self.log_manager = LogManager(self.opt)
        self.data_manager = DataManager(self.opt, self.log_manager)
        self.model_manager = ModelManager(self.opt, self.log_manager)
        self.train_manager = TrainManager(self.opt, self.log_manager)
        self.log_manager.init_state(None)
        self.data_manager.init_state(None)
        self.model_manager.init_state(None)
        self.build_network()

        atexit.register(self.atexit)

    def run(self):
        self.log_manager.print('server process has been started')
        self.create_clients()
        self.train_clients()

    def build_network(self) -> object:
        self.global_model = self.model_manager.build_resnet18_decomposed()
        num_connected = self.opt.num_clients
        self.restored_clients = {i:self.model_manager.build_resnet18_decomposed() for i in range(num_connected)}


    def create_clients(self):
        opt_copied = copy.deepcopy(self.opt)
        self.log_manager.print('creating client processes on cpu ... ')
        num_parallel = self.opt.num_clients
        self.clients = {i:Client(i, opt_copied) for i in range(num_parallel)}

    def train_clients(self):
        start_time = time.time()

        args = self.opt
        cids = np.arange(self.opt.total_num_clients).tolist()
        num_connected = self.opt.num_clients
        whole_episode = self.opt.whole_episode

        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)

        if not os.path.exists(args.states_dir):
            os.mkdir(args.states_dir)


        for curr_round in range(whole_episode):  # curr_round：0~599

            self.loss = 0.0
            self.clients_num = 0
            self.curr_round = curr_round

            for i in range(0, 95, num_connected):  # training packets
                self.updates = []

                new_list = cids[i:i + num_connected]
                if len(new_list) !=num_connected:
                    continue
                self.log_manager.print('training clients (round:{}, lr:{}, connected:{})'.format(curr_round + 1, self.opt.lr, new_list))
                self.clients_num += len(new_list)
                print('------------{}/{}------------'.format(curr_round + 1, whole_episode))
                while len(new_list)>0:

                    for gpu_id, gpu_client in self.clients.items():
                        if len(new_list) == 0:
                            break
                        else:
                            net = self.clients[gpu_id].local_model
                            global_model_state_dict_copy = self.global_model.encoder.state_dict().copy()
                            net.encoder.load_state_dict(global_model_state_dict_copy)

                            cid = new_list.pop(0)
                            with tf.device('/device:GPU:{}'.format(gpu_id)):
                                self.invoke_client(gpu_client, cid, curr_round, net)

                for t in range(len(self.updates)):
                    self.loss += self.updates[t][1]

                if i == 90 and (curr_round+1)%2 ==0:
                    loss = self.loss / self.clients_num
                    print('_____________________________________________last_loss={}'.format(loss))

                aggregated_encoder = self.aggregate(self.updates)
                self.global_model.encoder.load_state_dict(copy.deepcopy(aggregated_encoder.state_dict()))

            if self.opt.cos:
                self.opt.lr *= 0.5 * (1. + math.cos(math.pi * curr_round / 3000))

            if (curr_round+1)%2 ==0:
                total_top1, total_top5, total_num = test(self.global_model.encoder, curr_round, self.opt)
                acc_1 = total_top1 / total_num * 100
                print('After aggregated_test result-->total num:{},  average acc:{:.4f}%'.format(total_num, acc_1))
                acc_5 = total_top5 / total_num * 100
                print('After aggregated_test result-->total num:{},  average acc:{:.4f}%'.format(total_num, acc_5))

        self.log_manager.print('all clients done')
        self.log_manager.print('server done. ({}s)'.format(time.time()-start_time))
        sys.exit()

    def invoke_client(self, client, cid, curr_round, net):

        optimizer = torch.optim.SGD(net.encoder.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, momentum=0.9)

        update = client.train_one_round_ls(cid, curr_round, net, optimizer)

        self.updates.append(update)


    def aggregate(self, updates):
        aggregated_encoder = copy.deepcopy(updates[0][0].encoder)

        total = sum(update[2] for update in updates)
        total_neg = sum((total - update[2]) for update in updates)

        for param in aggregated_encoder.parameters():
            param.data.zero_()

        for i in range(len(updates)):
            weight = (total - updates[i][2]) / total_neg

            model_params = updates[i][0].encoder.parameters()

            for agg_param, param in zip(aggregated_encoder.parameters(), model_params):
                agg_param.data += weight * param.data
        return aggregated_encoder


    def atexit(self):
        for thrd in self.threads:
            thrd.join()
        self.log_manager.print('all client threads have been destroyed.' )
