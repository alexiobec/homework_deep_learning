import numpy as np
import torch
import torch.nn as nn
import math as m
import plotly.graph_objects as go
import sys
import pandas as pd

h = 10
nepochs = 1
dn = 50.
nout = 1
ker = 7

train = np.array(pd.read_csv('meteo/2019.csv').values.tolist())
trainx = torch.Tensor(
    [float(elmt[1])/dn for elmt in train][:-1]).view(1, -1, 1)
trainy = torch.Tensor([float(elmt[1])/dn for elmt in train][1:]).view(1, -1, 1)


test = np.array(pd.read_csv('meteo/2020.csv').values.tolist())
testx = torch.Tensor(
    [float(elmt[1])/dn for elmt in test[:-7]][:-1]).view(1, -1, 1)
testy = torch.Tensor(
    [float(elmt[1])/dn for elmt in test[7:]][1:]).view(1, -1, 1)

evaluation = torch.Tensor(
    [float(elmt[1])/dn for elmt in test[:92]][:-1]).view(1, -1, 1)
# trainx = 1, seqlen, 1
# trainy = 1, seqlen, 1
trainds = torch.utils.data.TensorDataset(trainx, trainy)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)
testds = torch.utils.data.TensorDataset(testx, testy)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)
crit = nn.MSELoss()


class Mod(nn.Module):
    def __init__(self):
        super(Mod, self).__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, 1, 7, stride=1),
                                 nn.Sigmoid())

    def forward(self, x):
        # x.shape = (1,351,7) -> (N,Hin,L)
        # cnn needs (N,L,Hin) (B,D,T)(batch,time,dim)
        xx = x.transpose(1, 2)
        y = self.cnn(xx)
        # y.shape = (1,1,346)
        return y.squeeze(0).squeeze(0)


def test(mod):
    mod.train(False)
    totloss, nbatch = 0., 0
    for data in testloader:
        inputs, goldy = data
        haty = mod(inputs)
        loss = crit(haty, goldy)
        totloss += loss.item()
        nbatch += 1
    totloss /= float(nbatch)
    mod.train(True)
    return totloss


def train(mod, nepochs, lr):
    optim = torch.optim.Adam(mod.parameters(), lr=lr)
    tab_train_loss = []
    tab_test_loss = []
    for epoch in range(nepochs):
        testloss = test(mod)
        totloss, nbatch = 0., 0
        for data in trainloader:
            inputs, goldy = data
            optim.zero_grad()
            haty = mod(inputs)
            loss = crit(haty, goldy)
            totloss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()
        totloss /= float(nbatch)
        tab_train_loss.append(totloss)
        tab_test_loss.append(testloss)
        # print("err", totloss, testloss)
    # print("fin", totloss, testloss, file=sys.stderr)
    return tab_train_loss, tab_test_loss


if __name__ == "__main__":
    tab_value = []
    tab_loss = []
    tab = [21]*100
    n_epochs = 50
    for _ in tab:
        mod = Mod()
        lr = 0.02
        nepochs = 35
        trainloss, testloss = train(mod, nepochs, lr)
        test(mod)
        tens = evaluation
        # print(mod(tens).squeeze())
        tab_value.append(mod(tens).squeeze()[-1].item()*50)
    mean = sum(tab_value)/len(tab_value)
    print(f"max value = {max(tab_value)}")
    print(f"min value = {min(tab_value)}")
    print(f"average value = {mean}")
    print(f"écart-type = {np.std(tab_value)}")
    print("real value = 21.0")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.array(tab_value), x=np.arange(100),
                             mode='markers', name="Prédiction du CNN"))
    fig.add_trace(go.Scatter(y=np.array([mean]*100), x=np.arange(100),
                             mode='lines', name="Moyenne du CNN"))
    fig.add_trace(go.Scatter(y=np.array(tab), x=np.arange(100),
                             mode='lines', name="Valeur réelle"))
    fig.update_layout(
        title=f"Prédiction de la température du 07/04/2020 par le CNN",
        xaxis_title="Exécution",
        yaxis_title="Température (°C)"
    )
    fig.show()
