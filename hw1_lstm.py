import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from collections import defaultdict

TRAIN_PATH = './topicclass/topicclass_train.txt'
TEST_PATH = './topicclass/topicclass_test.txt'
DEV_PATH = './topicclass/topicclass_valid.txt'

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))

pad = w2i["<pad>"]
start = w2i["<sos>"]
end = w2i["<eos>"]
UNK = w2i["<unk>"]

def read_dataset(filename):
  with open(filename, "r") as f:
    for line in f:
      tag, words = line.lower().strip().split(" ||| ")
      yield ([start] + [w2i[x] for x in words.split(" ")] + [end], t2i[tag])

# Read in the data
TRAIN = list(read_dataset(TRAIN_PATH))
DEV = list(read_dataset(DEV_PATH))
TEST = list(read_dataset(TEST_PATH))

i2t = {}
for key in t2i.keys():
  i2t[t2i[key]] = key

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('Using %s' % DEVICE)

nwords = len(w2i)
ntags = len(t2i)
lverbose = 200
BATCH_SIZE = 16*4
clip_value = 5
EPOCHS = 10
DEV_EPOCH = 1
TEST_EPOCH = 1
LOGDIR = './logs_' + time.strftime("%Y%m%d-%H%M%S")


class SentTrDataset(Dataset):
  def __init__(self, data):
    self.data = data
    self.len_ = len(self.data)

  def __len__(self):
    return self.len_

  def __getitem__(self, index):
    X = np.array(self.data[index][0])
    Y = np.array(self.data[index][1])

    Xt, Yt = torch.from_numpy(X).long(), torch.from_numpy(Y).long()
    return Xt, Yt


class SentTsDataset(Dataset):
  def __init__(self, data):
    self.data = data
    self.len_ = len(self.data)

  def __len__(self):
    return self.len_

  def __getitem__(self, index):
    X = self.data[index][0]

    Xt = torch.from_numpy(X).long()
    return Xt


def collate_lines(seq_list):
  inputs, targets = zip(*seq_list)
  lens = [len(seq) for seq in inputs]
  seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
  inputs = [inputs[i] for i in seq_order]
  targets = [targets[i] for i in seq_order]
  return inputs, targets


def collate_lines_test(seq_list):
  inputs = seq_list
  lens = [len(seq) for seq in inputs]
  seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
  inputs = [inputs[i] for i in seq_order]
  return inputs, seq_order


class Net(nn.Module):
  def __init__(self, nwords, num_classes):
    super(Net, self).__init__()

    self.embedding = nn.Embedding(nwords, 400*3)
    self.rl = nn.LeakyReLU(inplace=True)

    self.lstm = nn.LSTM(input_size=400*3, hidden_size=256, num_layers=5, bidirectional=True, bias=False, dropout=0.4)
    self.l1 = nn.Linear(256*2, 256*2, bias=False)
    self.l2 = nn.Linear(256*2, num_classes, bias=False)

  def forward(self, packed_input):
    input_padded, input_lens = rnn.pad_packed_sequence(packed_input, batch_first=True)
    embed_padded = self.embedding(input_padded)
    lstm_packed = rnn.pack_padded_sequence(embed_padded.transpose(0, 1), input_lens)

    hidden = None
    output_packed, hidden = self.lstm(lstm_packed, hidden)
    output_padded, _ = rnn.pad_packed_sequence(output_packed)
    x = self.rl(self.l1(torch.mean(output_padded.transpose(0, 1), 1)))
    x = self.l2(x)
    return x


def validation(net, loader, epoch):
  net.eval()
  tp = 0.0
  ct = 0.0
  fname = 'dev_predictions.txt' 
  with open(fname, 'w') as f:
    for seq_list, labels in loader:
      Xt = rnn.pack_sequence(seq_list)
      Xt = Xt.to(DEVICE)

      logits = net(Xt).cpu()

      ct += len(labels)
      for i in range(len(labels)):
        f.write(i2t[torch.argmax(logits[i]).item()])
        f.write('\n')
        if labels[i] == torch.argmax(logits[i]):
          tp = tp + 1

  print('Epoch[%d] Accuracy: %f' % (epoch, tp/ct))


def test_model(net, loader):
  net.eval()
  tp = 0.0
  ct = 0.0
  fname = 'test_predictions.txt' 
  with open(fname, 'w') as f:
    for seq_list, seq_order in loader:
      Xt = rnn.pack_sequence(seq_list)
      Xt = Xt.to(DEVICE)

      logits = net(Xt).cpu()

      for i in range(logits.size(0)):
        f.write(i2t[torch.argmax(logits[i]).item()])
        f.write('\n')


def initialize_weights(net):
  for mod in net.modules():
    if isinstance(mod, (nn.Conv2d, nn.Linear)):
      torch.nn.init.kaiming_normal_(mod.weight)


def main(filepath=None, test=False):
  train_dataset = SentTrDataset(TRAIN)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                            collate_fn=collate_lines)
  dev_dataset = SentTrDataset(DEV)
  dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                          collate_fn=collate_lines)

  ts_dataset = SentTsDataset(TEST)
  ts_loader = DataLoader(ts_dataset, batch_size=1, shuffle=False, num_workers=4,
                         collate_fn=collate_lines_test)

  net = Net(nwords, ntags).float()
  if DEVICE == 'cuda':
    net = net.cuda()

  if filepath is None:
     initialize_weights(net)

  criterion = nn.CrossEntropyLoss()
  optimizer_ADAM = optim.Adam(net.parameters(), lr=0.001)

  if filepath is not None:
    state = torch.load(filepath)
    net.load_state_dict(state['state_dict'])
    if test:
      print('Dumping Test submission.')
      with torch.no_grad():
        validation(net, dev_loader, 100)
        test_model(net, ts_loader)
      return

  for epoch in range(EPOCHS):
    net.train()
    epoch_loss = 0.0
    for i, (seq_list, Yt) in enumerate(train_loader):
      Xt = rnn.pack_sequence(seq_list)
      Xt = Xt.to(DEVICE)
      Yt = torch.stack(Yt).to(DEVICE)

      logits = net(Xt)

      optimizer_ADAM.zero_grad()
      loss = criterion(logits, Yt)
      loss.backward()
      torch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
      optimizer_ADAM.step()

      epoch_loss += loss.item()
      if (i+1) % lverbose == 0:
        print('Epoch[%d][%d] loss: %f' % (epoch+1, i+1, epoch_loss/lverbose))
        epoch_loss = 0.0

    if (epoch + 1) % DEV_EPOCH == 0:
      with torch.no_grad():
        validation(net, dev_loader, epoch+1)

    if (epoch + 1) % TEST_EPOCH == 0:
      #with torch.no_grad():
      #  test_model(net, decoder, ts_loader, epoch+1)
      state = {
               'state_dict': net.state_dict()
              }
      torch.save(state, './' + time.strftime("%Y%m%d-%H%M%S") + '_' + str(epoch + 1) + '.dat')


  state = {
           'state_dict': net.state_dict()
          }
  torch.save(state, './' + time.strftime("%Y%m%d-%H%M%S") + '_' + str(epoch + 1) + '.dat')


if __name__ == '__main__':
  main(*sys.argv[1:])
