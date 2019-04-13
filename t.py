import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation
import random

dataset = np.load('../dataset/wiki.train.npy')
fixtures_pred = np.load('../fixtures/prediction.npz')  # dev
fixtures_gen = np.load('../fixtures/generation.npy')  # dev
fixtures_pred_test = np.load('../fixtures/prediction_test.npz')  # test
fixtures_gen_test = np.load('../fixtures/generation_test.npy')  # test
vocab = np.load('../dataset/vocab.npy')


NUM_EPOCHS = 10
BATCH_SIZE = 1
SEQ_LEN=200
EMBED_SIZE=200
HIDDEN_SIZE=50
NLAYERS=1

def t():
    # shuffle all sequences
    np.random.shuffle(dataset)

    # concatenate your articles and build into batches
    all_together = np.concatenate(dataset)

    n_seq = all_together.shape[0] // SEQ_LEN

    all_together = all_together[:n_seq * SEQ_LEN]

    batch_data = all_together.reshape(-1, BATCH_SIZE, SEQ_LEN)

    for n in range(batch_data.shape[0]):
        txt = batch_data[n]
        print("step ", n)
        yield (torch.from_numpy(txt[:, :-1]), torch.from_numpy(txt[:, 1:]))

cut=random.randint((0,SEQ_LEN))
print(cut)
