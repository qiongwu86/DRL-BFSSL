from par import Parser
from utils.misc import *
from modules.data_generator import DataGenerator
import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def main(opt):
    
    if opt.job == 'data':
        opt = set_data_config(opt)
        dg = DataGenerator(opt)
        dg.generate_data()

    elif opt.job == 'train':
        opt = set_config(opt)

        from models.fedmatch.server import Server
        server = Server(opt)
        server.run()


def set_config(opt):
    
    os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu

    if opt.results_dir == '':
        opt.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-BFSSL")

    if opt.states_dir == '':
        opt.states_dir = './states'

    opt = set_data_config(opt)

    return opt

def set_data_config(opt):
    opt.mixture_fname = 'saved_mixture.npy'
    if 'c10' in opt.task:
        opt.datasets = [0]
        opt.num_classes = 10

    elif 'fmnist' in opt.task:
        opt.datasets = [4]
        opt.num_classes = 10

    opt.total_num_clients = 95  # training packets
    opt.num_epochs_client = 2
    opt.ptr = 0

    if 'biid' in opt.task or 'bimb' in opt.task:
        opt.sync = False
        opt.num_tasks = 1
        opt.whole_epoch = 200

    elif 'simb' in opt.task:
        opt.sync = True
        opt.num_tasks = 10
        opt.whole_epoch = 200

    else:
        print('no correct task was given: {}'.format(opt.task))
        os._exit(0)
    return opt

if __name__ == '__main__':
    main(Parser().parse())
