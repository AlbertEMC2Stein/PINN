import numpy as np
import torch
from torch import tensor
from torch.autograd.functional import jacobian, hessian
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('macosx')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n = 128
        self.inL = nn.Linear(1, n)
        self.h1L = nn.Linear(n, n)
        self.h2L = nn.Linear(n, n)
        self.h3L = nn.Linear(n, n)
        self.oL = nn.Linear(n, 1)

    def forward(self, x):
        activation = torch.tanh
        y = activation(self.inL(x))
        y = activation(self.h1L(y))
        y = activation(self.h2L(y))
        y = activation(self.h3L(y))
        y = activation(self.oL(y))
        return y


def plot_nn():
    ts = torch.linspace(0, 5, 1001).reshape((1001, 1))

    plt.plot(ts, ts * torch.exp(-ts), color='r', label="Analytical solution")
    plt.plot(ts, u(ts).detach(), label="Numerical solution")
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
        "EPOCHS": 1000,
        "SAMPLES": 25,
        "LR": 0.01,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
    }

    data = {
        "initial bias": 5,
        "initial": tensor([0.]),
        "target": tensor([0.]),
        "inputs": torch.linspace(0, 5, settings["SAMPLES"], requires_grad=True).reshape((settings["SAMPLES"], 1))
    }

    u = Net()
    opt = torch.optim.Adam(u.parameters(), lr=settings["LR"])
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.85)

    def f(x):
        # u'' - µ(1 - u^2)u' + u = 0, µ >> 0
        # result = hessian(u, x) - 10 * (1 - u(x).pow(2)) * jacobian(u, x) + u(x)

        # u' = -u => u(x) = exp(-x)
        # result = jacobian(u, x) + u(x)

        # u' = exp(-x) - u => u(x) = xexp(-x)
        result = jacobian(u, x) + u(x) - torch.exp(-x)

        # u' = -2xu => u(x) = exp(-x^2)
        # result = jacobian(u, x) + 2 * x * u(x)

        # u'' = -u => u(x) = cos(x)
        # result = u(x) + hessian(u, x)

        return result

    def loss(dict):
        criterion = torch.nn.MSELoss()

        f_out = f(dict["inputs"][0])
        for x in dict["inputs"][1:]:
            f_out = torch.cat((f_out, f(x)))

        E_0 = criterion(u(dict["initial"]), dict["target"])
        E_f = criterion(f_out, torch.zeros_like(dict["inputs"]))

        return dict["initial bias"] * E_0 + E_f

    trainTemplate = "\r[INFO] epoch: {:04d} train loss: {:.4f} learning rate: {:.4f}"
    losses = []
    for epoch in range(settings["EPOCHS"]):
        u.train()

        def closure():
            opt.zero_grad()
            err = loss(data)
            err.backward()
            return err

        losses += [closure().item()]
        print(trainTemplate.format(epoch + 1, losses[-1], scheduler.get_last_lr()[0]), end="")

        opt.step(closure)
        scheduler.step()

    plot_nn()
    plot_loss()
