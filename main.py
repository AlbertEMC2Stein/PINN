import numpy as np
import torch
from torch.autograd.functional import jacobian
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('macosx')


def smooth_LeakyReLU(x, a=0.02, b=15):
    return (1 + a) / 2 * x + (1 - a) / (2 * b) * torch.sqrt((b * x).pow(2) + 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n = 512
        self.inL = nn.Linear(2, n)
        self.h1L = nn.Linear(n, n)
        self.oL = nn.Linear(n, 1)

    def forward(self, x):
        activation = smooth_LeakyReLU
        y = activation(self.inL(x))
        y = activation(self.h1L(y))
        y = self.oL(y)
        return y


def diff(func, x, wrt):
    def unwrap(tuple):
        result = []
        for i, n in enumerate(tuple):
            result += [i] * n

        return result

    def jrec(jf, x, list):
        if not list:
            return jf(x)

        else:
            return jacobian(lambda _x: jrec(jf, _x, list[1:]), x, create_graph=True, vectorize=True)[list[0]]

    return jrec(func, x, unwrap(wrt)).item()


def neuraldiff(model, inputs, wrt):
    outputs = model(inputs).shape[-1]

    results = torch.tensor([[]])
    for i in range(outputs):
        temp = []
        for x in inputs:
            temp += [diff(lambda _x: model(_x)[i], x, wrt)]

        temp = torch.tensor([temp])
        results = torch.cat((results, temp), dim=-1 if i == 0 else 0)

    return results.t()


def load_numerical_data(file_path, noise_factor):
    data = np.genfromtxt(file_path, delimiter=',')
    data += noise_factor * np.random.uniform(-1, 1, size=data.shape)

    return torch.tensor(data)


def generate_training_data(xtrange, initial_f, boundary_fs, sample_sizes):
    data = {}
    ts = torch.zeros(sample_sizes["initial"], dtype=torch.float32)
    xs = torch.tensor(np.random.uniform(*xtrange[0:2], sample_sizes["initial"]), dtype=torch.float32)
    txs = torch.stack((ts, xs)).t()

    data["initial"] = (txs, initial_f(xs))

    for i, boundary_f in enumerate(boundary_fs):
        ts = torch.tensor(np.random.uniform(*xtrange[2:], sample_sizes["boundary"]), dtype=torch.float32)
        xs = torch.tensor(np.random.choice(xtrange[0:2], sample_sizes["boundary"]), dtype=torch.float32)
        txs = torch.stack((ts, xs)).t()

        data["boundary_%d" % i] = (txs, boundary_f(txs))

    ts = torch.tensor(np.random.uniform(*xtrange[2:], sample_sizes["collocation"]), dtype=torch.float32)
    xs = torch.tensor(np.random.uniform(*xtrange[0:2], sample_sizes["collocation"]), dtype=torch.float32)
    txs = torch.stack((ts, xs)).t()

    data["collocation"] = (txs, torch.zeros(txs.shape[0]))

    return data


def pick_samples(data, n):
    idx = np.random.choice(range(len(data)), n, replace=False)
    return data[idx]


def plot_nn():
    xs = torch.linspace(-1, 1, 100)
    ts = torch.linspace(0, 1, 100)
    t, x = torch.meshgrid(ts, xs, indexing='xy')
    tx = torch.stack((t, x), dim=2)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(t, x, u(tx).reshape(t.shape).detach())
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()


def plot_loss():
    plt.plot(losses, color='r')
    plt.xlim([0, settings["EPOCHS"] - 1])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == "__main__":
    settings = {
        "SAMPLE_SIZES": {
            "initial": 10,
            "boundary": 10,
            "collocation": 200
        },
        "EPOCHS": 200,
        "LR": 0.001,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
    }

    dict = generate_training_data([-1, 1, 0, 1],
                                  lambda x: -torch.sin(torch.pi * x),  # x.pow(2) * torch.cos(torch.pi * x)
                                  [lambda tx: torch.zeros(tx.shape[0])],
                                  settings["SAMPLE_SIZES"]
                                  )

    u = Net()
    opt = torch.optim.Rprop(u.parameters(), lr=settings["LR"])
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.8)

    def f(x):
        # uₜ = (0.01/π)uₓₓ - uuₓ
        result = neuraldiff(u, x, (1, 0)) + u(x) * neuraldiff(u, x, (0, 1)) - \
                 0.01 / torch.pi * neuraldiff(u, x, (0, 2))

        # uₜ = 0
        result = neuraldiff(u, x, (1, 0))

        # uₜ = 5(u(1 - |u|) + u³ - u)
        # result = neuraldiff(u, x, (1, 0)) - 5 * (u(x) * (1 - abs(u(x))) + u(x).pow(3) - u(x))

        # uₜ = 0.0001uₓₓ - 5u² + 5u
        # result = neuraldiff(u, x, (1, 0)) - 0.0001 * neuraldiff(u, x, (0, 2)) - 5 * u(x).pow(3) + 5 * u(x)

        return result

    def loss(dict):
        criterion = torch.nn.MSELoss()
        split = 20

        u0 = u(dict["input0"])
        ub = u(dict["inputb"])
        # ub = neuraldiff(u, dict["inputb"], (0, 1))

        tgt0 = dict["target0"].reshape(u0.shape)
        tgtb = dict["targetb"].reshape(ub.shape)
        # tgtb = torch.zeros_like(ub)

        fi = torch.zeros_like(dict["inputi"])
        if epoch % 100 >= split:
            fi = f(dict["inputi"])

        E_u0 = criterion(u0, tgt0.reshape(u0.shape))
        E_ub = criterion(ub, tgtb.reshape(ub.shape))
        E_fi = criterion(fi, torch.zeros_like(fi))

        return E_u0 + E_ub + E_fi

    trainTemplate = "\r[INFO] epoch: {} train loss: {:.4f} learning rate: {:.8f}"
    losses = []
    for epoch in range(settings["EPOCHS"]):
        trainLoss = 0
        samples = 0
        u.train()

        data = {
            "input0": dict["initial"][0],
            "target0": dict["initial"][1],
            "inputb": dict["boundary_0"][0],
            "targetb": dict["boundary_0"][1],
            "inputi": pick_samples(dict["collocation"][0], 100)
        }

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
