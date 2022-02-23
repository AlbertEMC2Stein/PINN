import numpy as np
import torch
from torch.autograd.functional import jacobian
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('macosx')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n = 128
        self.inL = nn.Linear(2, n)
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
        return y  # torch.tensor(-torch.sin(torch.pi * x[..., 1]), requires_grad=True).reshape(y.shape)


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
            "initial": 5,
            "boundary": 5,
            "collocation": 100
        },
        "EPOCHS": 1000,
        "LR": 0.01,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
    }

    u = Net()
    opt = torch.optim.Adam(u.parameters(), lr=settings["LR"])
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

    def f(x):
        # u_t = (0.01/tf.pi)*u_xx - u*u_x
        result = neuraldiff(u, x, (1, 0)) + u(x) * neuraldiff(u, x, (0, 1)) - \
                 0.01 / torch.pi * neuraldiff(u, x, (0, 2))

        # u_t = 0
        # result = neuraldiff(u, x, (1, 0))

        return result

    def loss(dict):
        criterion = torch.nn.MSELoss()
        u0 = u(dict["input0"])
        tgt0 = dict["target0"].reshape(u0.shape)

        ub = u(dict["inputb"])
        tgtb = dict["targetb"].reshape(ub.shape)

        fi = torch.zeros_like(dict["inputi"])
        if epoch % 25 <= 4:
            fi = f(dict["inputi"])

        E_u0 = criterion(u0, tgt0.reshape(u0.shape))
        E_ub = criterion(ub, tgtb.reshape(ub.shape))
        E_fi = criterion(fi, torch.zeros_like(fi))

        return E_u0 + E_ub + E_fi

    trainTemplate = "\r[INFO] epoch: {} train loss: {:.4f} learning rate: {:.4f}"
    losses = []
    for epoch in range(settings["EPOCHS"]):
        trainLoss = 0
        samples = 0
        u.train()

        data = generate_training_data([-1, 1, 0, 1],
                                      lambda x: -torch.sin(torch.pi * x),
                                      [lambda tx: torch.zeros(tx.shape[0])],
                                      settings["SAMPLE_SIZES"]
                                      )

        data = {
            "input0": data["initial"][0],
            "target0": data["initial"][1],
            "inputb": data["boundary_0"][0],
            "targetb": data["boundary_0"][1],
            "inputi": data["collocation"][0]
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
