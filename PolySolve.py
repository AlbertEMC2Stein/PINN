import torch
import torch.nn as nn
from torch import tensor, exp, sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macosx')


def p(x, coeffs, n):
    def factor(j):
        result = 1
        for i in range(n):
            result *= j - i

        return result

    if n >= len(coeffs):
        return torch.zeros_like(x)

    return sum([c * factor(i) * x ** (i-n) for i, c in enumerate(coeffs) if i-n >= 0])


class Model(nn.Module):
    def __init__(self, *fixed, n=10):
        super().__init__()
        self.fixed = list(fixed)
        self.weights = nn.Parameter(torch.rand(n - len(fixed)))

    def forward(self, x):
        return p(x, self.coeffs, 0)

    @property
    def coeffs(self):
        return torch.cat([tensor(self.fixed), self.weights])


def solve(epochs):
    x = torch.linspace(0, 1, 2001)
    loss_history = []
    trainTemplate = "\r[INFO] epoch: {:04d} train loss1: {:.4f} learning rate: {:.8f}"

    for epoch in range(epochs):
        f1.train()
        f2.train()

        def closure():
            opt.zero_grad()
            loss = to_zero()
            loss.backward()
            return loss

        loss_history += [closure().item()]
        print(trainTemplate.format(epoch + 1, loss_history[-1], opt.param_groups[0]['lr']), end="")

        opt.step(closure)
        scheduler.step(loss_history[-1])

        if epoch % 10 == 0:
            plt.plot(x, f1(x).detach())
            plt.pause(0.0001)
            plt.clf()

    plt.show()
    return loss_history


if __name__ == '__main__':
    f1 = Model(1, 0, n=25)
    f2 = Model(0, n=25)

    def to_zero():
        criterion = torch.nn.SmoothL1Loss()
        x = torch.linspace(0, 1, 501, requires_grad=True)

        # y'' - µ(1 - y^2)y' + y = 0
        # y = p(x, f.coeffs, 2) / (4 * pi**2) + p(x, f.coeffs, 0)

        # y' + 2xy = 0
        # y = p(x, f.coeffs, 1) / 3 + 2 * x * p(x, f.coeffs, 0)

        # y'' + y = 0  ==>  (y1, y2)' = (y2, -y1)
        y1 = p(x, f1.coeffs, 1) - 10 * p(x, f2.coeffs, 0)
        y2 = p(x, f2.coeffs, 1) + 10 * (0.2 * p(x, f2.coeffs, 0) + p(x, f1.coeffs, 0))

        # (y1, y2)' = (A + y1^2 y2 - (B + 1)y1, By1 - y1^2 y2)
        # y1 = p(x, f1.coeffs, 1) - 10 * (1 + p(x, f1.coeffs, 0).pow(2) * p(x, f2.coeffs, 0) - 4 * p(x, f1.coeffs, 0))
        # y2 = p(x, f2.coeffs, 1) - 10 * (3 * p(x, f1.coeffs, 0) - p(x, f1.coeffs, 0).pow(2) * p(x, f2.coeffs, 0))

        # y'' - µ(1 - y^2)y' + y = 0  ==>  (y1, y2)' = (y2, µ(1 - y1^2)y2 - y1)
        # y1 = p(x, f1.coeffs, 1) - 12 * p(x, f2.coeffs, 0)
        # y2 = p(x, f2.coeffs, 1) - 12 * (5 * (1 - p(x, f1.coeffs, 0).pow(2)) * p(x, f2.coeffs, 0) - p(x, f1.coeffs, 0))

        return criterion(y1, torch.zeros_like(y1)) + criterion(y2, torch.zeros_like(y2))

    params = iter([f1.weights, f2.weights])
    opt = torch.optim.LBFGS(params, lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=10)

    losses = solve(500)

    plt.plot(losses, color='r')
    plt.show()


