import argparse

SEED = 1
NGT = 5
JOB = 'train'  # data or train
Server_EPOCH = 1
BASE_NETWORK = 'resnet18'
DATASET_PATH = './dataset'  # <-- set path
MODEL = 'BFSSL'
NC = 5
Whole_epoch = 200


class Parser:#类

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='FL_Train MoCo on CIFAR-10')
        self.set_arguments()
       
    def set_arguments(self):
        self.parser.add_argument('-g', '--gpu', default='0', type=str, help='to set gpu ids to use e.g. 0,1,2,...')
        self.parser.add_argument('-gc', '--gpu-clients', type=str, help='to set number of clients per gpu e.g. 3,3,3,...')
        self.parser.add_argument('-m', '--model', default=MODEL, type=str, help='to set model to experiment')
        self.parser.add_argument('-t', '--task', type=str, default="biid-c10", help='to set task to experiment')
        self.parser.add_argument('-n', '--num-clients', default=NC, type=float, help='to set fraction of clients per round')  #选中车辆的比例
        self.parser.add_argument('-E', '--whole_epoch', type=str, default=Whole_epoch, help='to set whole epoch')


        self.parser.add_argument('-j', '--job', type=str, default=JOB, help='to set job to execute e.g. data, train, test, etc.')
        self.parser.add_argument('-ngt', '--num-gt', type=int, default=NGT, help='to set num ground truth per class')
        self.parser.add_argument('-e', '--experiment', type=str, help='to set experiment name')
        self.parser.add_argument('--base-network', type=str, default=BASE_NETWORK, help='to set base networks alexnet-like, etc.‘resnet9//resnet18’')
        self.parser.add_argument('--task-path', type=str, default=DATASET_PATH, help='to set task path')
        self.parser.add_argument('--seed', type=int, default=SEED, help='to set fixed random seed')

        # lr: 0.06 for batch 512 (or 0.03 for batch 256)
        self.parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
        self.parser.add_argument('--epochs', default=Server_EPOCH, type=int, metavar='N', help='number of total epochs to run')
        self.parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
        self.parser.add_argument('--cos', default=True, action='store_true', help='use cosine lr schedule')

        self.parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

        self.parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
        self.parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
        self.parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

        self.parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

        self.parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

        # knn monitor
        self.parser.add_argument('--knn-k', default=100, type=int, help='k in kNN monitor')
        self.parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

        # utils
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
        self.parser.add_argument('--states-dir', default='', type=str, metavar='PATH',
                                 help='path to load model (default: none)')

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        return args
