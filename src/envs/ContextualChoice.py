import torch
import numpy as np


class ContextualChoiceTask():

    def __init__(self, vec_dim, T, t_noise_off=5):
        self.vec_dim = vec_dim
        self.T = T
        self.t_noise_off = t_noise_off
        assert t_noise_off < T

    def gen_data(
            self, n_examples,
            combine_xk=True, to_torch=True, device='cpu'
    ):
        evidences, cues, targets = self.sample(n_examples)
        # form the 2nd part of the data
        perm = np.random.permutation(len(evidences))
        evidences_p = np.array([evidences[i] for i in perm])
        targets_p = np.array([targets[i] for i in perm])
        cues_p = np.array([cues[i] for i in perm])
        # concat evidence and cues
        if combine_xk:
            evidences_cues = np.dstack([evidences, cues])
            evidences_cues_p = np.dstack([evidences_p, cues_p])
        # combine two parts of the data
        X = np.vstack([evidences_cues, evidences_cues_p])
        K = np.vstack([cues, cues_p])
        Y = np.vstack([targets, targets_p])
        # to pytorch form
        if to_torch:
            X = to_th(X)
            K = to_th(K)
            Y = to_th(Y, dtype=torch.LongTensor)
        return X, K, Y

    def sample(self, n_examples):
        evidences = [None] * n_examples
        cues = [None] * n_examples
        targets = [None] * n_examples
        for i in range(n_examples):
            evidences[i], cues[i], targets[i] = self._sample()
        return np.array(evidences), np.array(cues), np.array(targets)

    def _sample(self):
        """
        evidence:
            initially ambiguous,
            after `t_noise_off`, become predictive about the target
        """
        evidence = np.random.normal(
            loc=np.sign(np.random.normal()),
            size=(self.T, self.vec_dim)
        )
        # integration results
        target_value = 1 if np.sum(evidence) > 0 else 0
        target = np.tile(target_value, (self.T,))
        # corrupt the evidence input
        evidence[:self.t_noise_off] = np.random.normal(
            loc=0, size=(self.t_noise_off, self.vec_dim)
        )
        # generate a cue
        cue_ = np.random.normal(size=(self.vec_dim, ))
        cue = np.tile(cue_, (self.T, 1))
        return evidence, cue, target


def to_th(np_array, dtype=torch.FloatTensor, device='cpu'):
    return torch.from_numpy(np_array).type(dtype).to(device)
