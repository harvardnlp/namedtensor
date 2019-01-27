from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from namedtensor import NamedTensor, ndistributions, ntorch

parser = argparse.ArgumentParser(description="VAE MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="enables CUDA training",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data", train=True, download=True, transform=transforms.ToTensor()
    ),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data", train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = x.op(self.fc1, F.relu)
        return h1.op(self.fc21, z="x"), h1.op(self.fc22, z="x")

    def reparameterize(self, mu, logvar):
        normal = ndistributions.Normal(mu, logvar.exp())
        return normal.rsample(), normal

    def decode(self, z):
        return z.op(self.fc3, F.relu).op(self.fc4, x="z").sigmoid()

    def forward(self, x):
        mu, logvar = self.encode(x.stack(x=("ch", "height", "width")))
        z, normal = self.reparameterize(mu, logvar)
        return self.decode(z), normal


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, var_posterior):
    BCE = recon_x.reduce2(
        x.stack(h=("ch", "height", "width")),
        lambda x, y: F.binary_cross_entropy(x, y, reduction="sum"),
        ("batch", "x"),
    )
    prior = ndistributions.Normal(
        ntorch.zeros(dict(batch=1, z=1)), ntorch.ones(dict(batch=1, z=1))
    )
    KLD = ndistributions.kl_divergence(var_posterior, prior).sum()
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        data = NamedTensor(data, ("batch", "ch", "height", "width"))
        optimizer.zero_grad()
        recon_batch, normal = model(data)
        loss = loss_function(recon_batch, data, normal)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            data = NamedTensor(data, ("batch", "ch", "height", "width"))
            recon_batch, normal = model(data)
            test_loss += loss_function(recon_batch, data, normal).item()
            if i == 0:
                n = min(data.size("batch"), 8)
                group = [
                    data.narrow("batch", 0, n),
                    recon_batch.split(
                        x=("ch", "height", "width"), height=28, width=28
                    ).narrow("batch", 0, n),
                ]

                comparison = ntorch.cat(group, "batch")
                save_image(
                    comparison.values.cpu(),
                    "results/reconstruction_" + str(epoch) + ".png",
                    nrow=n,
                )

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        test(epoch)
        train(epoch)

        with torch.no_grad():
            sample = ntorch.randn(dict(batch=64, z=20)).to(device)
            sample = model.decode(sample).cpu()
            save_image(
                sample.split(
                    x=("ch", "height", "width"), height=28, width=28
                ).values,
                "results/sample_" + str(epoch) + ".png",
            )
