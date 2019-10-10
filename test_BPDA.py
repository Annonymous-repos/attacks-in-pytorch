import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
from attacks import BPDAattack
from util import _save_image, ensure_dir, accu
from advertorch.defenses import MedianSmoothing2D, BitSqueezing, JPEGFilter
from torchvision import transforms


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


import torchvision.datasets as datasets

MNIST_PATH = 'data/'


def get_mnist_test_loader(batch_size, shuffle=False):
    return torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_PATH, train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, 5),
            nn.MaxPool2d(kernel_size=2),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(50 * 16, 500),
            nn.ReLU())
        self.linear2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.Block(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        logits = self.linear2(x)
        return logits


class whitebox(object):

    def __init__(self, args):
        self.args = args

        # setup data_loader instances
        self.data_loader = get_mnist_test_loader(100, shuffle=False)

        # setup device
        self.device = torch.device('cuda:' + args.device)

        # build defense architecture
        bits_squeezing = BitSqueezing(bit_depth=5)
        median_filter = MedianSmoothing2D(kernel_size=3)
        jpeg_filter = JPEGFilter(10)

        self.defense = nn.Sequential(
            jpeg_filter,
            bits_squeezing,
            median_filter,
        )
        # build classifier architecture
        self.oracle = LeNet()
        self.oracle = self.oracle.to(self.device)
        self.oracle.load_state_dict(torch.load(args.resume_oracle))
        self.oracle.eval()

        self.adversary = BPDAattack(self.oracle, self.defense, self.device, epsilon=0.3,
                                    learning_rate=0.5,
                                    max_iterations=100,
                                    )

    def eval_(self):
        """
        :return:
        """
        total_metrics = 0
        defense_metrics = 0
        for batch_idx, (gt_image, label) in enumerate(tqdm(self.data_loader)):
            gt_image, label = gt_image.to(self.device), label.to(self.device)

            adv = self.adversary.generate(gt_image, label)
            adv = adv.detach()

            logits = self.oracle(adv)
            total_metrics += accu(logits, label)

            reformed = self.defense(adv)
            logits = self.oracle(reformed)
            defense_metrics += accu(logits, label)

        total_metrics /= len(self.data_loader)
        defense_metrics /= len(self.data_loader)

        return total_metrics, defense_metrics, adv, gt_image, reformed


def main(args):
    ensure_dir(args.results_dir)
    wbx = whitebox(args)

    total_accu, defense_aacu, adv, gt, reform = wbx.eval_()
    _save_image(args.results_dir, adv, 'adv')
    _save_image(args.results_dir, gt, 'gt')
    _save_image(args.results_dir, reform, 'reform')
    print(total_accu)
    print(defense_aacu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('--resume_oracle', default='LeNet.pth', type=str,
                        help='path to latest checkpoint of oracle (default: None)')

    parser.add_argument('--results_dir', default='results', type=str,
                        help='output dictionary')

    parser.add_argument('--device', default='0', type=str,
                        help='output dictionary')

    args = parser.parse_args()

    main(args)
