from src.networks import AutoEncoder
from src.common import LinearDataset
from src import MinkowskiModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(12)
np.random.seed(12)

N = 5000
x = np.zeros((N, 9))
psi = np.random.uniform(0, np.pi/2, size=N)
phi = np.random.uniform(0, np.pi/2, size=N)
hi = np.random.uniform(0, np.pi/2, size=N)
t = np.random.uniform(0, np.pi/2, size=N)

x[:, 0] = np.cos(psi)*np.cos(phi)*np.cos(hi)*np.cos(t)
x[:, 1] = np.cos(psi)*np.sin(phi)*np.cos(hi)*np.cos(t)
x[:, 2] = np.sin(psi)*np.cos(hi)*np.cos(t)
x[:, 3] = np.sin(hi)*np.cos(t)
x[:, 4] = np.sin(t)
x[:, 5] = np.cos(np.linspace(0, 1, N))
x[:, 6] = np.cos(np.linspace(0, 0.5, N))
x[:, 7] = np.cos(np.linspace(0.5, 1, N))
x[:, 8] = np.cos(np.linspace(0, 1, N))

# middle = 3 # x.shape[1]  # MinkowskiModel(verbose=True).fit_predict(x)[1]
# model = AutoEncoder(input_size=x.shape[1], middle_layer=middle)
models = [AutoEncoder(input_size=x.shape[1], middle_layer=i) for i in range(1, x.shape[1])]
names = [f'AutoEncoder({x.shape[1]}, {i})' for i in range(1, x.shape[1])]

test_size = int(x.shape[0] * 0.33)  # get holdout part
train = LinearDataset(x=x[:-test_size], ranges=[], device=models[0].device)
test = LinearDataset(x=x[-test_size:], ranges=[], device=models[0].device)
criterion = nn.MSELoss()
train_loader = DataLoader(train, batch_size=100, shuffle=False)

batch_losses = {}
test_losses = {}
for idx, model in enumerate(tqdm(models)):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch_losses[names[idx]] = []
    for epoch in range(50):
        sum_loss = []
        for batch_id, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            sum_loss.append(loss.item())
        batch_losses[names[idx]].append(np.mean(sum_loss))
    test_loss = []
    for j in range(len(test)):
        model.eval()
        response = model(test[j])
        loss = criterion(response, test[j])
        test_loss.append(loss.detach().numpy())
    test_losses[names[idx]] = np.mean(test_loss)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(211)
for key, value in batch_losses.items():
    ax.plot(value, label=key)

ax = fig.add_subplot(212)
for i, (key, value) in enumerate(test_losses.items()):
    ax.bar(i, value, label=key)

# add xlabel, ylabel
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

plt.tight_layout()
plt.savefig('middle_layer_comparison.png', dpi=600, quality=100)

plt.show()
