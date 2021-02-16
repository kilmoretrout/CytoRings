import numpy as np
import matplotlib.pyplot as plt
from func import solve_harmonic_series, compute_harmonic, get_A, get_C, get_A_prime
from scipy.interpolate import interp1d

import torch
from torch.nn import Linear, Conv1d
from torch.autograd import Variable

import itertools


def differentiator(f, N = 7):
    if N == 7:
        ret = []
        for k in range(3, len(f) - 3):
            ret.append((39*(f[k + 1] - f[k - 1]) + 12*(f[k + 2] - f[k - 2]) - 5*(f[k + 3] - f[k - 3])) / (96))

        return ret
    elif N == 11:
        ret = []
        for k in range(5, len(f) - 5):
            ret.append((322*(f[k + 1] - f[k - 1]) + 256*(f[k + 2] - f[k - 2]) + 39*(f[k + 3] - f[k - 3]) - 32*(f[k + 4] - f[k - 4]) - 11*(f[k + 5] - f[k-5])) / (1536))

        return ret

class ShapeModel(torch.nn.Module):
    def __init__(self, A, A_prime, C, N = 3):
        super(ShapeModel, self).__init__()

        self.lin = Linear(2*N + 1, A.shape[0], bias = False)
        self.lin.weight.data = torch.FloatTensor(A)

        self.lin_prime = Linear(2*N + 1, A.shape[0], bias = False)
        self.lin_prime.weight.data = torch.FloatTensor(A_prime)

        self.t_prime = Conv1d(A.shape[0], A.shape[0], kernel_size = C.shape[2], bias = False, padding = C.shape[2] // 2)
        self.t_prime.weight.data = torch.FloatTensor(C)

    def forward(self, x):
        # get the coordinate
        x_pos = self.lin(x)

        # get the derivative
        dx = self.lin_prime(x)

        # get the spatial derivative
        dt = self.t_prime(x_pos.transpose(1, 0).view(1, x_pos.shape[1], x_pos.shape[0])).squeeze().transpose(1, 0)

        return x_pos, dx, dt

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        return

    def forward(self, x, dx, dxt, y, dy, dyt, target_x, target_y, only_dot = False):
        norm_dx = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        norm_dxt = torch.sqrt(torch.pow(dxt, 2) + torch.pow(dyt, 2))

        xx = target_x - x
        yy = target_y - y

        dot_loss = torch.mean(torch.pow(xx*dx + yy*dy, 2))
        recon_loss = torch.mean(torch.pow(torch.sqrt(torch.pow(x - target_x, 2) + torch.pow(y - target_y, 2)), 2))

        if only_dot:
            return dot_loss
        else:
            return dot_loss + recon_loss

# get a perturbed version of a cone
t = np.linspace(0., 100., 101)
print(t[1] - t[0])

r = 110. - t

theta = np.linspace(-np.pi, np.pi, 72)

X = []
X_theta = []

Nk = 3

for k in range(len(t)):
    x = []
    y = []

    x_ = r[k]*np.cos(theta)
    y_ = r[k]*np.sin(theta)

    for j in range(100):
        x.extend(list(x_ + np.random.normal(0, 10., 72)))
        y.extend(list(y_ + np.random.normal(0, 10., 72)))

    X_ = np.vstack([np.array(x, dtype = np.float32), np.array(y, dtype = np.float32)]).T
    theta_ = np.arctan2(X_[:,1], X_[:,0])

    X.append(X_)
    X_theta.append(theta_)

print('created arrays...')

co = []
xy = []

xy_real = []

N = 1024

for k in range(len(X)):
    xy_ = X[k]
    theta_ = X_theta[k]

    ix = [np.argmin(np.abs(theta_ - u)) for u in np.linspace(-np.pi, np.pi, N)]

    x_co = solve_harmonic_series(xy_[ix,0], theta_[ix], N = Nk)
    y_co = solve_harmonic_series(xy_[ix,1], theta_[ix], N = Nk)

    xy_real.append(xy_[ix])

    co.append(np.vstack([x_co, y_co]))

    x_new = compute_harmonic(theta, x_co, N = Nk)
    y_new = compute_harmonic(theta, y_co, N = Nk)

    xy.append(np.array([x_new, y_new]).T)

xy = np.array(xy)

dx = np.array(differentiator(xy[:,:,0]))
dy = np.array(differentiator(xy[:,:,1]))

error = (t[1] - t[0] - np.sqrt(dx**2 + dy**2)).flatten()

co = np.array(co)

xy_real = np.array(xy_real)
target_x = torch.FloatTensor(xy_real[:, :, 0])
target_y = torch.FloatTensor(xy_real[:, :, 1])

A = get_A(np.linspace(-np.pi, np.pi, N), N = Nk)
A_prime = get_A_prime(np.linspace(-np.pi, np.pi, N), N = Nk)
C = get_C(accuracy = 4, n_out = N)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ShapeModel(A, A_prime, C, N = Nk)
model.to(device)

model.eval()

target_x = target_x.to(device)
target_y = target_y.to(device)

count = 0

alpha = list(np.logspace(0., 1., 100)*np.max(xy[0,:,0])*np.max(xy[0,:,1])*4)
beta = list(np.logspace(0., 1., 100)*np.max(xy[0,:,0])*np.max(xy[0,:,1])*4)

todo = itertools.product(alpha, beta)

Z = np.zeros((100, 100))
Z[::] = np.nan

f = np.array([1., 1., 1.])

co_x = co[:, 0, :]
co_y = co[:, 1, :]

co_x = torch.tensor(torch.from_numpy(co_x).float(), requires_grad = True, device = device)
co_y = torch.tensor(torch.from_numpy(co_y).float(), requires_grad = True, device = device)

optimizer = torch.optim.Adam([co_x, co_y], lr = 0.01)
criterion = Loss()

loss = np.inf

factors = np.random.uniform(0., 1000, (4,))

count = 0
ix = 0

# do both
while True:
    optimizer.zero_grad()

    x, dx, dxt = model.forward(co_x)
    y, dy, dyt = model.forward(co_y)

    L = criterion(x, dx, dxt, y, dy, dyt, target_x, target_y)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    L.backward()
    optimizer.step()

    if L.item() < loss:
        loss = L.item()
        print(L.item())

        co_x_new = co_x.detach().cpu().numpy()
        co_y_new = co_y.detach().cpu().numpy()

        count = 0
    else:
        count += 1

        if count > 1000:
            break


    ix += 1

xy_new = []

for k in range(len(co_x)):
    co_x_ = co_x_new[k]
    co_y_ = co_y_new[k]

    x_new = compute_harmonic(theta, co_x_, N=Nk)
    y_new = compute_harmonic(theta, co_y_, N=Nk)

    xy_new.append(np.array([x_new, y_new]).T)

xy_new = np.array(xy_new)

dx = np.array(differentiator(xy_new[:, :, 0]))
dy = np.array(differentiator(xy_new[:, :, 1]))

fig, axes = plt.subplots(nrows=2, ncols=2)

for k in range(xy.shape[0]):
    axes[0, 0].plot(xy[k, :, 0], xy[k, :, 1])
for k in range(xy.shape[1]):
    axes[0, 1].plot(xy[:, k, 0], xy[:, k, 1])

for k in range(xy.shape[0]):
    axes[1, 0].plot(xy_new[k, :, 0], xy[k, :, 1])
for k in range(xy.shape[1]):
    axes[1, 1].plot(xy_new[:, k, 0], xy[:, k, 1])

plt.show()
plt.close()

dx = np.array(differentiator(xy_new[:, :, 0]))
dy = np.array(differentiator(xy_new[:, :, 1]))

new_error = (t[1] - t[0] - np.sqrt(dx ** 2 + dy ** 2)).flatten()

plt.imshow(t[1] - t[0] - np.sqrt(dx ** 2 + dy ** 2))
plt.colorbar()
plt.show()
plt.close()

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].hist(error, bins=35)
axes[1].hist(new_error, bins=35)

plt.show()
plt.close()

















    

    



    

    
