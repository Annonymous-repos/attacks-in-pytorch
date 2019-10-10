import argparse
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn

MNIST_PATH = 'data/'


def get_mnist_train_loader(batch_size, shuffle=True):
    return torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_PATH, train=True, download=True,
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


def main(args):
    device = torch.device('cuda:' + args.device)
    data_loader = get_mnist_train_loader(100)

    loss_fn = nn.CrossEntropyLoss()

    model = LeNet()
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for batch_idx, (gt_image, label) in enumerate(data_loader):
            gt_image, label = gt_image.to(device), label.to(device)
            optimiser.zero_grad()
            logits = model(gt_image)
            loss = loss_fn(logits, label)
            loss.backward()
            optimiser.step()
    torch.save(model.state_dict(), 'LeNet.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-d', '--device', default='0', type=str,
                        help='config file path (default: None)')

    args = parser.parse_args()

    main(args)
