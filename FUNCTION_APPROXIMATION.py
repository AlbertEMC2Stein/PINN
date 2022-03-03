import torch
import time
from torch import tensor
import torch.nn as nn
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n = 512
        self.il = nn.Linear(1, n, bias=True)
        self.ol = nn.Linear(n, 1, bias=True)

    def forward(self, x):
        activation = torch.nn.LeakyReLU()
        y = activation(self.il(x))
        y = self.ol(y)
        return y


def approximate(f):
    def loss():
        xs = torch.linspace(*xrange, settings["SAMPLES"], requires_grad=True).reshape((settings["SAMPLES"], 1))
        return torch.nn.MSELoss()(u(xs), f(xs))

    trainTemplate = "\r[INFO] epoch: {:04d} train loss: {:.4f} learning rate: {:.4f}"

    losses = []
    for epoch in range(settings["EPOCHS"]):
        u.train()

        def closure():
            opt.zero_grad()
            err = loss()
            err.backward()
            return err

        losses += [closure().item()]
        print(trainTemplate.format(epoch + 1, losses[-1], opt.param_groups[0]['lr']), end="")

        opt.step(closure)
        scheduler.step(losses[-1])

        if losses[-1] < 1e-6:
            break

    return losses


def plot_nn():
    xs = torch.linspace(*xrange, 1001).reshape((1001, 1))

    plt.plot(xs, f(xs), color='r', label="Analytical solution")
    plt.plot(xs, u(xs).detach(), label="Numerical solution")
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def plot_loss():
    plt.plot(losses, color='r')
    plt.xlim([0, settings["EPOCHS"] - 1])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == "__main__":
    settings = {
        "EPOCHS": 200,
        "SAMPLES": 50,
        "LR": 0.01,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
    }

    xrange = [-3, 3]

    fs = [
        lambda x: torch.exp(-x),
        lambda x: torch.sin(x),
        lambda x: torch.cos(x),
        lambda x: torch.tanh(x),
        lambda x: torch.exp(-x.pow(2)),
        lambda x: torch.exp(-x) * torch.sin(x),
        lambda x: torch.exp(-1 / x.pow(2))
    ]

    for f in fs:
        device = torch.device(settings["DEVICE"])
        u = Net().to(device)
        opt = torch.optim.LBFGS(u.parameters(), lr=settings["LR"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.95, patience=5)

        losses = approximate(f)

        plot_nn()
        plot_loss()

        time.sleep(0.5)
