
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

from src import MinkowskiModel
from src.common import LinearDataset
from src.networks import LinearNet
from sklearn.datasets import make_regression

np.random.seed(11)
torch.manual_seed(25)

N = 3125
x = np.zeros((N, 10))
psi = np.random.uniform(0, np.pi/2, size=N)
phi = np.random.uniform(0, np.pi/2, size=N)
hi = np.random.uniform(0, np.pi/2, size=N)
t = np.random.uniform(0, np.pi/2, size=N)

x[:, 0] = np.cos(psi)*np.cos(phi)*np.cos(hi)*np.cos(t)
x[:, 1] = np.cos(psi)*np.sin(phi)*np.cos(hi)*np.cos(t)
x[:, 2] = np.sin(psi)*np.cos(hi)*np.cos(t)
x[:, 3] = np.sin(hi)*np.cos(t)
x[:, 4] = np.sin(t)
for j in range(5, 10):
    x[:, j] = np.cos(np.linspace(0, 1, N))


model = MinkowskiModel(eps_start=0.3, eps_edge_size=0.1, eps_end=0.4, distance_metric='chebyshev', f_steps=20, verbose=True)
main_coordinates = model.primary_dimensions(x)
print(main_coordinates)

model = LinearNet(input_size=len(main_coordinates), output_size=x.shape[1] - len(main_coordinates))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

test_size = int(x.shape[0] * 0.33)  # get holdout part
train = LinearDataset(x=x, target_idx=main_coordinates, device=model.device)
test = LinearDataset(x=x, target_idx=main_coordinates, device=model.device)


train_loader = DataLoader(train, batch_size=2048, shuffle=True)
test_loader = DataLoader(test, batch_size=1, shuffle=False)
mean_batch_loss = []

for epoch in range(500):
    avg_loss = []
    for batch_id, (train_x, train_y) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        out = model(train_y)
        # get only unknown columns
        loss = criterion(out, train_x)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
    mean_batch_loss.append(np.mean(avg_loss))
    print(f'epoch {epoch} loss: {mean_batch_loss[-1]}')

test_loss = 0.0
with torch.set_grad_enabled(False):
    for i, (test_x, test_y) in enumerate(test_loader):
        model.eval()
        response = model(test_y)
        loss = criterion(response, test_x)
        test_loss = loss.item()


plt.plot(mean_batch_loss)
plt.xlabel('epoch')
plt.ylabel('MSE Loss')
plt.title(f'MSE loss on holdout part = {test_loss:.4f}')
plt.savefig('decoder_results.png', dpi=600)
plt.show()
