import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

from src import MinkowskiModel
from src.common import LinearDataset
from src.networks import LinearNet


np.random.seed(11)
torch.manual_seed(25)

data = datasets.load_diabetes()
x = data.data
y = data.target

x = MinMaxScaler().fit_transform(x)


print(np.min(x, axis=0))
print(np.max(x, axis=0))


# find primary dimensions
main_coordinates = [0, 3]  # MinkowskiModel(verbose=True, f_steps=20, eps_edge_size=0.01).primary_dimensions(x)
print(main_coordinates)
model = LinearNet(input_size=len(main_coordinates), output_size=x.shape[1] - len(main_coordinates))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

test_size = int(x.shape[0] * 0.33)  # get holdout part
train = LinearDataset(x=x, target_idx=main_coordinates, device=model.device)
test = LinearDataset(x=x, target_idx=main_coordinates, device=model.device)


train_loader = DataLoader(train, batch_size=32, shuffle=True)
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

test_loss = 0.0
with torch.set_grad_enabled(False):
    for i, (test_x, test_y) in enumerate(test_loader):
        model.eval()
        response = model(test_y)
        loss = criterion(response, test_x)
        test_loss = loss.item()


plt.plot(mean_batch_loss)
plt.xlabel('epoch')
plt.ylabel('MSE * 100')
plt.title(f'MSE on holdout part = {test_loss:.2f}')
plt.savefig('decoder_results.png', dpi=600)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(411)
ax.scatter(x[:, 0], x[:, 3], c=y)
ax.set_xlim((0., 1.))
ax.set_ylim((0., 1.))

ax1 = fig.add_subplot(412)

x_tensor = torch.tensor(x[:, [0, 3]], dtype=torch.float, device=model.device)
x_new = model(x_tensor).cpu().detach().numpy()
ax1.scatter(x_new[:, 1], x_new[:, 0], c=y)
ax1.set_xlim((0., 1.))
ax1.set_ylim((0., 1.))


ax3 = fig.add_subplot(413)
ax3.plot(x[:, 0], label='x')
ax3.plot(x_new[:, 0], label='x_nex')

ax4 = fig.add_subplot(414)
ax4.plot(x[:, 3], label='x')
ax4.plot(x_new[:, 1], label='x_nex')

plt.legend()
plt.tight_layout()
plt.savefig('two_dimensions_results.png', dpi=600, quality=100)
plt.show()
