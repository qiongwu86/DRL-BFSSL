from modules.log_manager import LogManager
from modules.data_manager import DataManager
from modules.model_manager import ModelManager
from modules.train_manager import TrainManager

from MoCo_singleGPU_copy import *
import cv2

class Client:

    def __init__(self, gid, opt):
        self.opt = opt
        self.state = {'gpu_id': gid}

        self.log_manager = LogManager(self.opt)
        self.data_manager = DataManager(self.opt, self.log_manager)
        self.model_manager = ModelManager(self.opt, self.log_manager)
        self.train_manager = TrainManager(self.opt, self.log_manager)
        self.init_model()

    def init_model(self):
        self.local_model = self.model_manager.build_resnet18_decomposed()
        self.local_speed = 0

        self.log_manager.print('networks have been built')


    def init_state(self, client_id):
        self.state['client_id'] = client_id
        self.state['done'] = False
        self.state['task_names'] = []



    def train_one_round_ls(self, client_id, net, train_optimizer):

        self.log_manager.init_state(client_id)
        self.data_manager.init_state(client_id)
        self.model_manager.init_state(client_id)
        self.train_manager.init_state(client_id)
        self.init_state(client_id)

        self.load_data()

        client_v = self.get_speed()

        if client_v > 100:
            x_unlabeled_selected = self.x_unlabeled[:100]
            kernel_size = int(client_v * 0.2)
            angle = 90

            blurriness = int(client_v * 0.5)

            kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1

            # 生成运动模糊核
            kernel = np.zeros((kernel_size + blurriness, kernel_size + blurriness), dtype=np.float32)
            theta = np.radians(angle)
            center = (kernel_size + blurriness - 1) / 2

            for i in range(kernel_size):
                x = i - center
                y = -x  # 垂直方向
                x_new = x * np.cos(theta) - y * np.sin(theta) + center
                y_new = x * np.sin(theta) + y * np.cos(theta) + center
                kernel[int(y_new), int(x_new)] = 1.0 / kernel_size

            for n in range(len(x_unlabeled_selected)):
                image_selected = x_unlabeled_selected[n]
                # 使用滤波函数进行运动模糊
                self.x_unlabeled[n] = cv2.filter2D(image_selected, -1, kernel)


        for epoch in range(self.opt.num_epochs_client):

            x_unlabeled = self.x_unlabeled
            num_images = x_unlabeled.shape[0]
            shuffled_indices = np.random.permutation(num_images)
            x_batch = x_unlabeled[shuffled_indices]
            x_batch = x_batch[:512]

            self.local_model, self.log_manager.client_loss = train_client(net, x_batch, train_optimizer)

        return (self.local_model, self.log_manager.client_loss, client_v)

    def get_speed(self):
        random_value_1 = np.random.normal(0.5, 0.3)
        random_value = max(0, min(1, random_value_1))

        selected_value = (150 - 60) * random_value + 60
        client_v = round(selected_value / 10) * 10
        return client_v

    def init_new_task(self):
        self.state['curr_task'] += 1
        self.state['round_cnt'] = 0
        self.load_data()

    def load_data(self):

        self.x_unlabeled, self.y_unlabeled, task_name = self.data_manager.get_u_by_id(self.state['client_id'], 0)

        self.train_manager.set_task({
            'task_name':task_name.replace('u',''),
            'x_unlabeled':self.x_unlabeled,
            'y_unlabeled':self.y_unlabeled,
        })


    def get_task_id(self):
        return self.state['curr_task']

    def get_client_id(self):
        return self.state['client_id']

