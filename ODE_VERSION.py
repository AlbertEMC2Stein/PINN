import torch
from torch import tensor
from torch.autograd.functional import jacobian, hessian
import torch.nn as nn
import matplotlib.pyplot as plt


def smooth_LeakyReLU(x, a=0.02, b=15):
    return (1 + a) / 2 * x + (1 - a) / (2 * b) * torch.sqrt((b * x).pow(2) + 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n = 512
        self.il = nn.Linear(1, n, bias=True)
        self.ol = nn.Linear(n, 1, bias=True)

    def forward(self, x):
        activation = smooth_LeakyReLU  # torch.nn.LeakyReLU()
        y = activation(self.il(x))
        y = self.ol(y)
        return y


def gen_inputs(a, b, n):
    return torch.linspace(a, b, n).reshape((n, 1))


def calc_f(xs):
    f_out = f(xs[0])
    for x in xs[1:]:
        f_out = torch.cat((f_out, f(x)))

    return f_out


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

    plt.plot(ts, torch.exp(-ts.pow(2)), color='r', label="Analytical solution")
    plt.plot(ts, u(ts).detach(), label="Numerical solution")
    plt.scatter(data["inputs"], torch.zeros(settings["SAMPLES"]), c='black', s=0.5)
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


def plot_deviation():
    ts = torch.linspace(*xrange, 1001).reshape((1001, 1))

    plt.plot(ts, calc_f(ts).detach(), label="f")
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    settings = {
        "EPOCHS": 500,
        "SAMPLES": 201,
        "LR": 0.0001,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
    }

    xrange = [0, 2]
    data = {
        "initial bias": 1,
        "initial": tensor([[0.]]),
        "target": tensor([[1.]]),
        "inputs": gen_inputs(*xrange, settings["SAMPLES"])
    }

    device = torch.device(settings["DEVICE"])
    u = Net().to(device)
    opt = torch.optim.Rprop(u.parameters(), lr=settings["LR"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=5)

    def f(x):
        # u'' - µ(1 - u^2)u' + u = 0, µ >> 0
        # result = hessian(u, x) - 10 * (1 - u(x).pow(2)) * jacobian(u, x) + u(x)

        # u' = -u => u(x) = exp(-x)                             √ (mit u(0) = 1 und x in [0, 5])
        # result = jacobian(u, x) + u(x)

        # u' = exp(-x) - u => u(x) = xexp(-x)                   √ (mit u(0) = 0 und x in [0, 5])
        # result = jacobian(u, x) + u(x) - torch.exp(-x)

        # u' = -2xu => u(x) = exp(-x^2)                         √ (mit u(0) = 1 und x in [-2, 2])
        result = jacobian(u, x) + 2 * x * u(x)

        # u'' = -u => u(x) = cos(x)                             X
        # result = u(x) + hessian(u, x)

        # u' = 1 - u^2 => u(x) = tanh(x)                        √ (mit u(0) = 0 und x in [-2, 2])
        # result = jacobian(u, x) + u(x).pow(2) - 1

        return result

    losses = approximate()

    plot_nn()
    plot_deviation()
    plot_loss()
