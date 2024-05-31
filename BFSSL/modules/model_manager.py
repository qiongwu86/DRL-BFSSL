from MoCo_singleGPU_copy import *
from modules.model_layers import *

class ModelManager:
    def __init__(self, opt, log_manager):
        self.opt = opt
        self.log_manager = log_manager

    def init_state(self, client_id):
        self.state = {'client_id': client_id}

    def build_resnet18_decomposed(self):
        model = ModelMoCo(
            dim=self.opt.moco_dim,
            T=self.opt.moco_t,
            arch=self.opt.base_network,
            bn_splits=self.opt.bn_splits,
            symmetric=self.opt.symmetric,
        )
        return model


