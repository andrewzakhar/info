
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from src import MinkowskiModel
from src.common import LinearDataset
from src.networks import LinearNet
from sklearn.datasets import make_regression, make_friedman1

np.random.seed(11)
torch.manual_seed(25)


data = make_regression(n_samples=3125, n_features=10, n_informative=3, n_targets=1, random_state=11)

x = data[0]
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print(np.min(x, axis=0))
print(np.max(x, axis=0))

m = MinkowskiModel(distance_metric='chebyshev', f_steps=20, verbose=True)
main_coordinates = [3, 4, 6]  # model.primary_dimensions(x)
print(main_coordinates)

model = LinearNet(input_size=len(main_coordinates), output_size=x.shape[1] - len(main_coordinates))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

test_size = int(x.shape[0] * 0.33)  # get holdout part
train = LinearDataset(x=x[:test_size], target_idx=main_coordinates, device=model.device)
test = LinearDataset(x=x[test_size:], target_idx=main_coordinates, device=model.device)


train_loader = DataLoader(train, batch_size=2048, shuffle=True)
test_loader = DataLoader(test, batch_size=1, shuffle=False)
mean_batch_loss = []

for epoch in range(5000):
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


X = torch.tensor(x[:, [3, 4, 6]], device=model.device, dtype=torch.float)
Y = torch.tensor(x[:, [0, 1, 2, 5, 7, 8, 9]], device=model.device, dtype=torch.float)
model.eval()
result = model(X)
print(f'MSE = {criterion(result, Y)}')

numpy_result = result.cpu().detach().numpy()

x2 = x.copy()
for i, c in enumerate([0, 1, 2, 5, 7, 8, 9]):
    x2[:, c] = numpy_result[:, i]


x_embedded = TSNE(n_components=2).fit_transform(x)
x2_embedded = TSNE(n_components=2).fit_transform(x2)


fig = plt.figure()
ax = fig.add_subplot(121)
ax.scatter(x_embedded[:, 0], x_embedded[:, 1])
ax.grid(True)
ax.set_title('Real Data')

ax2 = fig.add_subplot(122)
ax2.scatter(x2_embedded[:, 0], x2_embedded[:, 1])
ax2.grid(True)
ax2.set_title('Reconstructed Data')

plt.savefig('TSNE_reg.png', dpi=600)
plt.show()

