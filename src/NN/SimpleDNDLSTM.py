"""
Implement a DND-based LSTM
- key is given by the input k_t
- memory content need to be learned
"""
import numpy as np
import torch
import torch.nn as nn
from NN.DND import DND

# constants
N_GATES = 4


class SimpleDNDLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, dict_len, memory_dim, kernel,
                 bias=True):
        super(SimpleDNDLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        # input-hidden weights
        self.i2h = nn.Linear(input_dim, (N_GATES+1) * hidden_dim, bias=bias)
        # hidden-hidden weights
        self.h2h = nn.Linear(hidden_dim, (N_GATES+1) * hidden_dim, bias=bias)
        # dnd
        self.dnd = DND(dict_len, memory_dim, kernel)
        # init
        self.reset_parameters()

    def reset_parameters(self):
        # neural network weight init
        std = 1.0 / np.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x_t, h, c, k_t):
        # unpack activity
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x_t = x_t.view(x_t.size(1), -1)
        k_t = k_t.view(1, -1)
        # transform the input info
        Wx = self.i2h(x_t)
        Wh = self.h2h(h)
        preact = Wx + Wh
        # get all gate values
        gates = preact[:, : N_GATES * self.hidden_dim].sigmoid()
        # split input(write) gate, forget gate, output(read) gate
        f_t = gates[:, :self.hidden_dim]
        i_t = gates[:, self.hidden_dim:2 * self.hidden_dim]
        o_t = gates[:, 2*self.hidden_dim:3 * self.hidden_dim]
        r_t = gates[:, -self.hidden_dim:]
        # stuff to be written to cell state
        c_t_new = preact[:, N_GATES * self.hidden_dim:].tanh()
        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(f_t, c) + torch.mul(i_t, c_t_new)
        # retrieve memory
        m_t = self.dnd.get_memory(k_t).tanh()
        # gate the memory; in general, can be any transformation of it
        h_t = torch.mul(o_t, c_t.tanh()) + torch.mul(r_t, m_t)
        # take a episodic snapshot
        self.dnd.save_memory(k_t, h_t)
        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        # output, in general, can be any diff-able transformation of h_t
        output_t = h_t
        # fetch activity
        cache = [f_t, i_t, o_t, r_t]
        return output_t, h_t, c_t, cache
