from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


class CIFAR10Pair(CIFAR10):
    """
    CIFAR10 Dataset.
    """
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)


train_transform1 = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
train_transform2 = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.reshape(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


class ModelBase(nn.Module):
    def __init__(self, feature_dim=128, arch=None, bn_splits=16):
        super(ModelBase, self).__init__()

        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        return x


class ModelMoCo(nn.Module):
    def __init__(self, dim=128, T=0.1, arch='resnet18', bn_splits=8, symmetric=True):  # k值存疑
        super(ModelMoCo, self).__init__()

        self.T = T
        self.symmetric = symmetric
        self.encoder = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        idx_shuffle = torch.randperm(x.shape[0])
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            k = self.encoder_k(im_k_)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long)

        my_loss = nn.CrossEntropyLoss()
        loss = my_loss(logits, labels)

        return loss, q, k

    def contrastive_loss_dual_temp(self, im_q, im_k):
        q = self.encoder(im_q)
        q = nn.functional.normalize(q, dim=1)

        k = self.encoder(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        l_neg = torch.einsum('nc,ck->nk', [q, k.T])
        mask_neg = torch.ones_like(l_neg, dtype=bool)
        mask_neg.fill_diagonal_(False)
        l_neg = l_neg[mask_neg].reshape(l_neg.size(0), l_neg.size(1) - 1)

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits_intra = logits / self.T
        prob_intra = F.softmax(logits_intra, dim=1)

        logits_inter = logits / (self.T * 10)
        prob_inter = F.softmax(logits_inter, dim=1)

        inter_intra = (1 - prob_inter[:, 0]) / (1 - prob_intra[:, 0])

        loss = -torch.nn.functional.log_softmax(logits_intra, dim=-1)[:, 0]
        loss = inter_intra.detach() * loss
        loss = loss.mean()

        return loss

    def forward(self, im1, im2):
        loss_12 = self.contrastive_loss_dual_temp(im1, im2)
        loss_21 = self.contrastive_loss_dual_temp(im2, im1)
        loss = (loss_12 + loss_21)/2.0

        return loss

def train_client(net, x_batch, train_optimizer):
    net.train()

    img1_all = torch.zeros([512, 3, 32, 32])
    img2_all = torch.zeros([512, 3, 32, 32])
    for i in range(len(x_batch[0])):
        img = x_batch[i]
        img = Image.fromarray(img)

        img1_all[i] = train_transform1(img)
        img2_all[i] = train_transform2(img)


    loss = net(img1_all, img2_all)
    print('loss.item={}'.format(loss.item()))

    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    return net, loss.item()


class CustomDataset(Dataset):
    def __init__(self, root):
        self.data = np.load(root, allow_pickle=True)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

memory_data = CIFAR10(root='data', train=True, download=True, transform=test_transform)  #
memory_loader = DataLoader(memory_data, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

test_data = CIFAR10(root='data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
def test(net, epoch, args):

    net.eval()
    classes = len(memory_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_loader, desc='Feature extracting'):
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.tensor(memory_loader.dataset.targets, device=feature_bank.device)

        test_bar = tqdm(test_loader)
        for data, target in test_bar:

            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

            top5_pred = pred_labels[:, :5]
            correct_top5 = top5_pred.eq(target.view(-1, 1).expand_as(top5_pred))
            correct_top5 = correct_top5.sum(dim=1)
            correct_top5 = correct_top5.sum().item()
            total_top5 += correct_top5

            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@5:{:.2f}%'.format(epoch, args.epochs, total_top5 / total_num * 100))

    return total_top1, total_top5, total_num