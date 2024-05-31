import time
import tensorflow as tf
import sys
sys.path.insert(0,'..')
from utils.misc import *
from third_party.mixture_loader.mixture import *

class DataGenerator:

    def __init__(self, opt):
        self.opt = opt
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
        self.load_third_party_data()

    def load_third_party_data(self):
        processed = os.path.join(self.opt.task_path, self.opt.mixture_fname)
        if os.path.exists(processed):
            print('loading mixture data: {}'.format(processed))
            self.mixture = np.load(processed, allow_pickle=True)
        else:
            print('downloading & processing mixture data')
            self.mixture,_,_ = get(base_dir=self.opt.task_path, fixed_order=True)
            np.save(processed, self.mixture)
        return 

    def get_dataset(self, dataset_id):
        print('load {} from third party ...'.format(self.did_to_dname[dataset_id]))
        self.dataset_id = dataset_id
        data = self.mixture.tolist()[0]
        x_train = data['train']['x']
        y_train = data['train']['y']

        x = np.concatenate([x_train])
        y = np.concatenate([y_train])

        x, y = self.shuffle(x, y)

        print('{}: {}'.format(self.did_to_dname[self.dataset_id], np.shape(x)))

        return x, y

    
    def generate_data(self):
        print('generating {} ...'.format(self.opt.task))
        start_time = time.time()
        self.task_cnt = -1 #
        self.is_imbalanced = True if 'imb' in self.opt.task else False
        self.is_streaming = True if 'simb' in self.opt.task else False
        for dataset_id in self.opt.datasets:
            x, y = self.get_dataset(dataset_id)  # 50000张，且打乱顺序
            self.generate_task(x, y)
        print('{} - done ({}s)'.format(self.opt.task, time.time()-start_time))

    def generate_task(self, x, y):
        u = self.split_train(x, y)
        self.save_task({
            'x': x,
            'y': y,
            'name': 'all_{}'.format(self.did_to_dname[self.dataset_id]),
            'labels': np.unique(y)
        })
        self.split_u(u)


    def split_train(self, x, y):
        data_by_label = {}
        self.labels = np.unique(y)
        for label in self.labels:
            idx = np.where(y[:]==label)[0]
            data_by_label[label] = {
                'x': x[idx],
                'y': y[idx]
            }

        self.num_u = 0
        u_by_label = {}
        for label, data in data_by_label.items():
            u_by_label[label] = {
                'x': data['x'][0:],
                'y': data['y'][0:]
            }
            self.num_u += len(u_by_label[label]['x'])
        print('num_u', self.num_u)

        return u_by_label
        

    def split_u(self, u):
        if self.is_imbalanced:
            z = np.random.dirichlet((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), size=10)
            for i in range(len(z)):
                sum = np.sum(z[i])
                for k in range(len(z)):
                    z[i][k] = z[i][k] / sum
            labels = list(u.keys())
            num_u_per_client = int(self.num_u/self.opt.total_num_clients)
            offset_per_label = {label:0 for label in labels}
            for cid in range(self.opt.total_num_clients):
                if self.is_streaming:
                    x_unlabeled = {tid:[] for tid in range(self.opt.num_tasks)}
                    y_unlabeled = {tid:[] for tid in range(self.opt.num_tasks)}
                    dist_type = cid%len(labels)
                    freqs = np.random.choice(labels, num_u_per_client, p=z[dist_type])
                    frq = []
                    for label, data in u.items():
                        num_instances = len(freqs[freqs==label])
                        frq.append(num_instances)
                        start = offset_per_label[label]
                        end = offset_per_label[label]+num_instances
                        _x = data['x'][start:end] 
                        _y = data['y'][start:end]
                        offset_per_label[label] = end 
                        num_instances_per_task = int(len(_x)/self.opt.num_tasks)
                        for tid in range(self.opt.num_tasks):
                            start = num_instances_per_task * tid
                            end = num_instances_per_task * (tid+1)
                            x_unlabeled[tid] = _x[start:end] if len(x_unlabeled[tid])==0 else np.concatenate([x_unlabeled[tid], _x[start:end]], axis=0)
                            y_unlabeled[tid] = _y[start:end] if len(y_unlabeled[tid])==0 else np.concatenate([y_unlabeled[tid], _y[start:end]], axis=0)
                    print('>>>> frq', frq)
                    for tid in range(self.opt.num_tasks):
                        x_task = x_unlabeled[tid]
                        y_task = y_unlabeled[tid]
                        x_task, y_task = self.shuffle(x_task, y_task)
                        self.save_task({
                            'x': x_task,
                            'y': tf.keras.utils.to_categorical(y_task, len(self.labels)),
                            'name': 'u_{}_{}_{}'.format(self.did_to_dname[self.dataset_id], cid, tid),
                            'labels': np.unique(y_task)
                        })
                else:
                    x_unlabeled = []
                    y_unlabeled = []
                    dist_type = cid%len(labels)
                    freqs = np.random.choice(labels, num_u_per_client, p=z[dist_type])
                    frq = []
                    for label, data in u.items():
                        num_instances = len(freqs[freqs==label])
                        frq.append(num_instances)
                        start = offset_per_label[label]
                        end = offset_per_label[label]+num_instances
                        x_unlabeled = data['x'][start:end] if len(x_unlabeled)==0 else np.concatenate([x_unlabeled, data['x'][start:end]], axis=0)
                        y_unlabeled = data['y'][start:end] if len(y_unlabeled)==0 else np.concatenate([y_unlabeled, data['y'][start:end]], axis=0)
                        offset_per_label[label] = end

                    x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)

                    self.save_task({
                        'x': x_unlabeled,
                        'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                        'name': 'u_{}_{}'.format(self.did_to_dname[self.dataset_id], cid),
                        'labels': np.unique(y_unlabeled)
                    })    
                    print('>>>> frq', frq)
        else:
            num_clients = 50000//520
            for cid in range(num_clients):
                x_unlabeled = []
                y_unlabeled = []
                for label, data in u.items():

                    num_unlabels_per_class = int(len(data['x'])/num_clients)
                    start = num_unlabels_per_class * cid
                    end = num_unlabels_per_class * (cid+1)
                    x_unlabeled = data['x'][start:end] if len(x_unlabeled)==0 else np.concatenate([x_unlabeled, data['x'][start:end]], axis=0)
                    y_unlabeled = data['y'][start:end] if len(y_unlabeled)==0 else np.concatenate([y_unlabeled, data['y'][start:end]], axis=0)

                x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)

                self.save_task({
                    'x': x_unlabeled,
                    'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                    'name': 'u_{}_{}'.format(self.did_to_dname[self.dataset_id], cid),
                    'labels': np.unique(y_unlabeled)
                })

    def save_task(self, data):
        save_task(base_dir=self.base_dir, filename=data['name'], data=data)
        print('filename:{}, labels:[{}], num_examples:{}'.format(data['name'],','.join(map(str, data['labels'])), len(data['x'])))

    def shuffle(self, x, y):
        idx = np.arange(len(x))
        random_shuffle(self.opt.seed, idx)
        return x[idx], y[idx]