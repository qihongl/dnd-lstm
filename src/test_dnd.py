"""demo: how to use dndlstm"""

import numpy as np
import torch
import torch.nn as nn
import time
from NN.DNDLSTM import DNDLSTM

#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_val = 0
torch.manual_seed(seed_val)
np.random.seed(seed_val)

"""
gen data: fit sin(x)
"""
n_time_steps = 60
# gen data
x_np = np.sin(np.linspace(0, 8*np.pi, n_time_steps))
x_np = np.reshape(x_np, newshape=(n_time_steps, 1, 1))
x = torch.from_numpy(x_np).type(torch.FloatTensor)

"""
model params
"""
# set lstm params
dim_input = 1
dim_hidden = 20
dim_output = dim_input
# dnd
dict_len = 60
kernel = 'cosine'
learning_rate = 1e-4
# train params
n_epochs = 100

"""
init the model
"""
# init model and hidden state.
lstm = DNDLSTM(dim_input, dim_hidden, dict_len, kernel)
readout = nn.Linear(dim_hidden, dim_output)
# initial state
h_0 = torch.zeros(1, 1, dim_hidden)
c_0 = torch.zeros(1, 1, dim_hidden)
# init optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    list(lstm.parameters())+list(readout.parameters()),
    lr=learning_rate
)


"""
train the model
"""

# train the model
clip_grad_norm_val = .5

# loop over epoch
losses = np.zeros(n_epochs,)
for i in range(n_epochs):
    time_start = time.time()

    # loop over time, for one training example
    for t, x_t in enumerate(x):
        # init rnn states
        if t == 0:
            h_t = h_0
            c_t = c_0
        # recurrent computation at time t
        out, h_t, c_t, _ = lstm(
            x_t.view(1, 1, -1), h_t, c_t)
        out = readout(out)
        # compute loss
        out_sqed = torch.squeeze(out, dim=0)
        loss = criterion(out_sqed, x_t)
        losses[i] += loss.item()

        # update weights
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(lstm.parameters(), clip_grad_norm_val)
        optimizer.step()

    # print out some stuff
    time_end = time.time()
    print(f'Epoch {i} | \t loss = {losses[i]}, \t time = {time_end - time_start}')
