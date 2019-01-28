"""
Example from  https://github.com/junwang4/CNN-sentence-classification-pytorch-2018
"""

from __future__ import print_function, division

import os
import sys
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold

import data_helpers
from namedtensor import NamedTensor, ntorch, nnn

# for obtaining reproducible results
np.random.seed(0)
torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
print("use_cuda = {}\n".format(use_cuda))

mode = "nonstatic"
mode = "static"
use_pretrained_embeddings = False
use_pretrained_embeddings = True

print("MODE      = {}".format(mode))
print(
    "EMBEDDING = {}\n".format(
        "pretrained" if use_pretrained_embeddings else "random"
    )
)

X, Y, vocabulary, vocabulary_inv_list = data_helpers.load_data()

vocab_size = len(vocabulary_inv_list)
sentence_len = X.shape[1]
num_classes = int(max(Y)) + 1  # added int() to convert np.int64 to int

print("vocab size       = {}".format(vocab_size))
print("max sentence len = {}".format(sentence_len))
print("num of classes   = {}".format(num_classes))

ConvMethod = "in_channel__is_embedding_dim"


class CNN(nn.Module):
    def __init__(
        self,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        embedding_dim=300,
        pretrained_embeddings=None,
    ):
        super(CNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.embedding = nnn.Embedding(vocab_size, embedding_dim).augment("h")
        self.embedding.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings)
        )
        self.embedding.weight.requires_grad = mode == "nonstatic"

        conv_blocks = []
        for kernel_size in kernel_sizes:
            conv1d = nnn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=1,
            )

            conv_blocks.append(conv1d)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.fc = nnn.Linear(num_filters * len(kernel_sizes), num_classes) \
                    .rename("h", "classes")
        self.dropout = nnn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).transpose("h", "slen")
        x_list = [
            conv_block(x).relu().max("slen")[0]
            for conv_block in self.conv_blocks
        ]
        out = ntorch.cat(x_list, "h")
        feature_extracted = out
        out = self.fc(self.dropout(out)).softmax("classes")
        return out, feature_extracted


def evaluate(model, x_test, y_test):
    inputs = NamedTensor(x_test, ("batch", "slen"))
    y_test = NamedTensor(y_test, ("batch",))
    preds, vector = model(inputs)
    preds = preds.max("classes")[1]
    eval_acc = (preds == y_test).sum("batch").item() / len(y_test)
    return eval_acc, vector.cpu().detach().numpy()


embedding_dim = 300
num_filters = 100
kernel_sizes = [3, 4, 5]
batch_size = 50


def load_pretrained_embeddings():
    pretrained_fpath_saved = os.path.expanduser(
        "models/googlenews_extracted-python{}.pl".format(
            sys.version_info.major
        )
    )
    if os.path.exists(pretrained_fpath_saved):
        with open(pretrained_fpath_saved, "rb") as f:
            embedding_weights = pickle.load(f)
    else:
        print("- Error: file not found : {}\n".format(pretrained_fpath_saved))
        print(
            '- Please run the code "python utils.py" to generate the file first\n\n'
        )
        sys.exit()

    # embedding_weights is a dictionary {word_index:numpy_array_of_300_dim}
    out = np.array(
        list(embedding_weights.values())
    )  # added list() to convert dict_values to a list for use in python 3
    # np.random.shuffle(out)

    print("embedding_weights shape:", out.shape)
    # pretrained embeddings is a numpy matrix of shape (num_embeddings, embedding_dim)
    return out


if use_pretrained_embeddings:
    pretrained_embeddings = load_pretrained_embeddings()
else:
    pretrained_embeddings = np.random.uniform(
        -0.01, -0.01, size=(vocab_size, embedding_dim)
    )


def train_test_one_split(cv, train_index, test_index):
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]

    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).long()
    dataset_train = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    x_test = torch.from_numpy(x_test).long()
    y_test = torch.from_numpy(y_test).long()

    model = CNN(
        kernel_sizes, num_filters, embedding_dim, pretrained_embeddings
    )
    if cv == 0:
        print("\n{}\n".format(str(model)))

    if use_cuda:
        model = model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.0002)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        tic = time.time()
        eval_acc, sentence_vector = evaluate(model, x_test, y_test)
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs, labels
            inputs = NamedTensor(inputs, ("batch", "slen"))
            labels = NamedTensor(labels, ("batch",))
            preds, _ = model(inputs)

            loss = preds.reduce2(labels, loss_fn, ("batch", "classes"))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        eval_acc, sentence_vector = evaluate(model, x_test, y_test)

        print(
            "[epoch: {:d}] train_loss: {:.3f}   acc: {:.3f}   ({:.1f}s)".format(
                epoch, loss.item(), eval_acc, time.time() - tic
            )
        )
    return eval_acc, sentence_vector


def do_cnn():
    cv_folds = 10
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
    acc_list = []
    tic = time.time()
    sentence_vectors, y_tests = [], []
    for cv, (train_index, test_index) in enumerate(kf.split(X)):
        acc, sentence_vec = train_test_one_split(cv, train_index, test_index)
        print(
            "cv = {}    train size = {}    test size = {}\n".format(
                cv, len(train_index), len(test_index)
            )
        )
        acc_list.append(acc)
        sentence_vectors += sentence_vec.tolist()
        y_tests += Y[test_index].tolist()
    print(
        "\navg acc = {:.3f}   (total time: {:.1f}s)\n".format(
            sum(acc_list) / len(acc_list), time.time() - tic
        )
    )

    # save extracted sentence vectors in case that we can reuse it for other purpose (e.g. used as input to an SVM classifier)
    # each vector can be used as a fixed-length dense vector representation of a sentence
    np.save("models/sentence_vectors.npy", np.array(sentence_vectors))
    np.save("models/sentence_vectors_y.npy", np.array(y_tests))


def main():
    do_cnn()


if __name__ == "__main__":
    main()
