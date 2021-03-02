import matplotlib.pyplot as plt
import numpy as np

from src.linear_models import LinePredictorCSI, get_loss

LOSS_THRESHOLD = 0.5
SPEED = 100

with open(f'..\\..\\csi_matrix\\data\\ChannelQriga_Freq_{SPEED}kmh_Scena0_test_3GPP_3D_UMa_NLOS_2p1GHz_numpy', 'rb') as file:
    csi_matrix = np.fromfile(file, dtype=np.complex).reshape(30, 300, 50, 64, 2)


csi_matrix.real *= 10 ** 6
csi_matrix.imag *= 10 ** 6

memory_in_use = 0
simulation_loss = 0
fitted_lines = []

# init first two
memory_in_use += csi_matrix[:, 0, :, :, :].nbytes
receiver_lp = LinePredictorCSI()
receiver_lp.fit(csi_matrix[:, :2, :, :, :])

memory_in_use += csi_matrix[:, 1, :, :, :].nbytes

fitted_lines.append((receiver_lp.fitted_lines_real, receiver_lp.fitted_lines_imag))

line_break = False
line_break_list = []

local_time = 2
t = 2
while t < 300:
    if not line_break:
        loss = get_loss(csi_matrix[:, t, :, :, :], local_time, receiver_lp.fitted_lines_real, receiver_lp.fitted_lines_imag, king='mpe')
        # print(loss)
        if t % 10 == 0 and t != 0:
            line_break = True
            line_break_list.append(t)
            # send data to receiver
            memory_in_use += csi_matrix[:, t, :, :, :].nbytes
        else:
            simulation_loss += loss
    else:
        # we already sent 1 dataitem on previous step so send another one and refit
        memory_in_use += csi_matrix[:, t, :, :, :].nbytes
        receiver_lp.fit(csi_matrix[:, t-1:t+1, :, :, :])
        line_break = False
        local_time = 0
    t += 1
    local_time += 1
    fitted_lines.append((receiver_lp.fitted_lines_real, receiver_lp.fitted_lines_imag))

print(f'Saved {memory_in_use / csi_matrix.nbytes:.2f}%')
print(f'Loss: {simulation_loss:.2f}% (not meaningful for mpe case)\n\tLoss to one TTI {(simulation_loss / csi_matrix.shape[1]):.4f}%')

# plot lines

base = np.full((30, 50, 64, 2), 1)
result_real = np.zeros((30, len(fitted_lines), 50, 64, 2))
result_imag = np.zeros((30, len(fitted_lines), 50, 64, 2))

t = 0
for i, line in enumerate(fitted_lines):
    if i in line_break_list:
        t = 0
    result_real[:, i, :, :, :] = line[0].k * (base + t) + line[0].b
    result_imag[:, i, :, :, :] = line[1].k * (base + t) + line[1].b
    t += 1
result = result_real + 1j * result_imag

# plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 4.5))

ax1.plot(csi_matrix[0, 1:, 0, 0, 0].imag, label='original data (imag)', linestyle='-', alpha=0.6, color='black', linewidth=2.5)
ax1.plot(result[0, :, 0, 0, 0].imag, label='adjusted data (imag)', color='red', linewidth=1)
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax2.plot(csi_matrix[0, 1:, 0, 0, 0].real, label='original data (real)', linestyle='-', alpha=0.6, color='black', linewidth=2.5)
ax2.plot(result[0, :, 0, 0, 0].real, label='adjusted data (real)', linestyle='-', color='red', linewidth=1)
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# ax1.set_title(f"""The simulation results for step size equal to 10 and csi matrix with
# high Ue speed ({SPEED} km/h). The average mpe is equal to {(simulation_loss / csi_matrix.shape[1]):.2f}%. Origin/Passed = {memory_in_use / csi_matrix.nbytes:.2f}""")
plt.savefig(f'simulation_example_with_constant_step_{SPEED}km.pdf', dpi=400)
plt.show()
