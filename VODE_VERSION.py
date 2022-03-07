import torch
from torch import tensor
from torch.autograd.functional import jacobian
import torch.nn as nn
import matplotlib.pyplot as plt


def smooth_LeakyReLU(x, a=0.02, b=15):
    return (1 + a) / 2 * x + (1 - a) / (2 * b) * torch.sqrt((b * x).pow(2) + 1)


class Net(nn.Module):
    def __init__(self, outs):
        super(Net, self).__init__()
        n = 256
        self.il = nn.Linear(1, n, bias=True)
        self.h1 = nn.Linear(n, n, bias=True)
        self.ol = nn.Linear(n, outs, bias=True)

    def forward(self, x):
        activation = smooth_LeakyReLU
        y = activation(self.il(x))
        y = activation(self.h1(y))
        y = self.ol(y)
        return y


def gen_inputs(a, b, n):
    return torch.linspace(a, b, n).reshape((n, 1))


def calc_f(xs):
    f_out = [f(xs[0])]
    for x in xs[1:]:
        f_out += [f(x)]

    return torch.stack(f_out)


def approximate():
    def loss(dict):
        f_out = calc_f(dict["inputs"])

        E_0 = torch.nn.MSELoss()(u(dict["initial"]), dict["target"])
        E_f = torch.nn.MSELoss()(f_out, torch.zeros_like(f_out))

        return dict["initial bias"] * E_0 + E_f

    trainTemplate = "\r[INFO] epoch: {:04d} train loss: {:.4f} learning rate: {:.8f}"

    losses = []
    for epoch in range(settings["EPOCHS"]):
        u.train()

        opt.zero_grad()
        err = loss(data)
        err.backward()

        losses += [err.item()]
        print(trainTemplate.format(epoch + 1, losses[-1], opt.param_groups[0]['lr']), end="")

        opt.step()
        scheduler.step(losses[-1])

    return losses


def plot_nn():
    ts = torch.linspace(*xrange, 1001).reshape((1001, 1))

    #plt.plot(ts, torch.cos(ts), color='r', label="Analytical solution")
    plt.plot(u(ts)[:, 0].detach(), u(ts)[:, 1].detach(), label="Numerical solution")
    #plt.scatter(data["inputs"], torch.zeros(settings["SAMPLES"]), c='black', s=0.5)
    plt.xlabel('y')
    plt.ylabel('y\'')
    plt.legend()
    plt.show()


def plot_loss():
    plt.plot(losses, color='r')
    plt.xlim([0, settings["EPOCHS"] - 1])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def plot_deviation():
    ts = torch.linspace(*xrange, 1001).reshape((1001, 1))

    plt.plot(ts, torch.norm(calc_f(ts), dim=-1).detach(), label="f")
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    settings = {
        "EPOCHS": 300,
        "SAMPLES": 25,
        "LR": 0.001,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
    }

    xrange = [0, 2]
    data = {
        "initial bias": 1,
        "initial": tensor([[0.]]),
        "target": tensor([[2., 0.]]),
        "inputs": gen_inputs(*xrange, settings["SAMPLES"])
    }

    device = torch.device(settings["DEVICE"])
    u = Net(2).to(device)
    opt = torch.optim.Rprop(u.parameters(), lr=settings["LR"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=5)

    def f(x):
        J = torch.squeeze(jacobian(u, x))
        U = u(x)

        # y'' = -y  ==>  (y, y')' = (y', -y)
        result = J - torch.tensor([U[1], -U[0]])
        # result = U - 2*torch.tensor([torch.cos(x), -torch.sin(x)])

        # y'' - µ(1 - y^2)y' + y = 0 ==> (y, y')' = (y', µ(1 - y^2)y' - y)
        # result = J - torch.tensor([U[1], 0 * (1 - U[0].pow(2)) * U[1] - U[0]]).reshape(J.shape)

        return result

    losses = approximate()

    plot_nn()
    plot_deviation()
    plot_loss()
