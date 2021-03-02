"""
 Dataset from https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from src import MinkowskiModel
from src.common import LinearDataset
from src.networks import LinearNet

np.random.seed(11)
torch.manual_seed(25)


data = pd.read_csv('CASP.csv')

x = data.to_numpy()

x = MinMaxScaler().fit_transform(x)

print(np.min(x, axis=0))
print(np.max(x, axis=0))

main_coordinates = [0, 1, 8]  # MinkowskiModel(verbose=True, f_steps=20, eps_edge_size=0.05).primary_dimensions(x)
# result [2, 3, 4] [0, 1, 8]
print(main_coordinates)

model = LinearNet(input_size=len(main_coordinates), output_size=x.shape[1] - len(main_coordinates))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.L1Loss()

test_size = int(x.shape[0] * 0.33)  # get holdout part
train = LinearDataset(x=x, target_idx=main_coordinates, device=model.device)
test = LinearDataset(x=x, target_idx=main_coordinates, device=model.device)


train_loader = DataLoader(train, batch_size=64, shuffle=True)
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
plt.ylabel('L1 loss')
plt.title(f'L1 loss on holdout part = {test_loss:.4f}')
# plt.savefig('decoder_results2.png', dpi=600)
plt.show()
