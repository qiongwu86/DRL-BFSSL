import glob
from utils.misc import *
import torch

class DataManager:

    def __init__(self, opt, log_manager):
        self.opt = opt
        self.log_manager = log_manager
        self.base_dir = os.path.join(self.opt.task_path, self.opt.task) 
        self.did_to_dname = {
            0: 'cifar10',
            1: 'cifar100',
            2: 'mnist',
            3: 'svhn',
            4: 'fashion_mnist',
            5: 'traffic_sign',
            6: 'face_scrub',
            7: 'not_mnist',
        }

    def init_state(self, client_id):
        self.state = {
            'client_id': client_id,
            'tasks': []
        }
        self.load_tasks()

    def load_state(self, client_id):
        self.state = np_load(os.path.join(self.opt.state_dir, '{}_data_manager.npy'.format(client_id))).item()

    def save_state(self):
        np_save(self.opt.state_dir, '{}_data_manager'.format(self.state['client_id']), self.state)

    def load_tasks(self):
        for d in self.opt.datasets:
            path = os.path.join(self.base_dir, self.did_to_dname[d]+'_'+str(self.state['client_id'])+'_*')
            self.tasks = [os.path.basename(p) for p in glob.glob(path)]
        self.tasks = sorted(self.tasks)


    def get_u_by_id(self, client_id, task_id):
        self.state['client_id'] = client_id
        for d in self.opt.datasets:
            path = os.path.join(self.base_dir, 'u_{}_{}*'.format(self.did_to_dname[d],str(self.state['client_id'])))
            self.tasks = sorted([os.path.basename(p) for p in glob.glob(path)])
        task = load_task(self.base_dir, self.tasks[task_id]).item()
        return task['x'], task['y'], task['name']


    def rescale(self, images):
        return torch.div(images, 255)

    def rescale_ndarray(self, images):
        return images.astype(np.float32)/255.
