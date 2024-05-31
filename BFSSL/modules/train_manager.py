import tensorflow as tf

from utils.misc import *
from MoCo_singleGPU_copy import *

loss = np.zeros(100, dtype=float)
freq = np.zeros(100, dtype=float)
class TrainManager:

    def __init__(self, opt, log_manager):
        self.opt = opt
        self.log_manager = log_manager

    def init_state(self, client_id):
        self.state = {
            'client_id': client_id,
            'total_num_params': 0
        }

    def load_state(self, client_id):
        self.state = np_load(os.path.join(self.opt.state_dir, '{}_train_manager.npy'.format(client_id))).item()

    def save_state(self):
        np_save(self.opt.state_dir, '{}_train_manager.npy'.format(self.state['client_id']), self.state)


    def train_one_round(self, curr_round, round_cnt, curr_task, train_optimizer, net):
        tf.keras.backend.set_learning_phase(True)
        self.state['curr_round'] = curr_round
        self.state['round_cnt'] = round_cnt
        self.state['curr_task'] = curr_task

        for epoch in range(self.params['num_epochs']):
            loss_s = 0
            self.state['curr_epoch'] = epoch
            self.num_confident = 0

            x_unlabeled = self.task['x_unlabeled']
            num_images = x_unlabeled.shape[0]
            shuffled_indices = np.random.permutation(num_images)

            x_batch = x_unlabeled[shuffled_indices]
            x_batch = x_batch[:512]

            batch_loss, new_net, k = train_client(self, net, x_batch, train_optimizer)


        return new_net, k, batch_loss

    def batch_shuffle(self, x):

        idx_shuffle = torch.randperm(x.shape[0])

        return x[idx_shuffle]


