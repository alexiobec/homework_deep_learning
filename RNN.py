import numpy as np
import torch
import torch.nn as nn
import math as m
import plotly.graph_objects as go
import sys
import pandas as pd

dn = 50.


train = np.array(pd.read_csv('meteo/2019.csv').values.tolist())
Ntrain = len(train)
trainxliste = []
for k in range(Ntrain-14):
    tab = train[k:k+7]
    newtab = [float(elmt[1])/dn for elmt in tab]
    trainxliste.append(newtab)
trainx = torch.Tensor(trainxliste).view(1, -1, 7)

trainyliste = []
for k in range(7, Ntrain-7):
    tab = train[k:k+7]
    newtab = [float(elmt[1])/dn for elmt in tab]
    trainyliste.append(newtab)
trainy = torch.Tensor(trainyliste).view(1, -1, 7)


test = np.array(pd.read_csv('meteo/2020.csv').values.tolist())
Ntest = len(test)
testxliste = []
for k in range(Ntest-14):
    tab = test[k:k+7]
    newtab = [float(elmt[1])/dn for elmt in tab]
    testxliste.append(newtab)
testx = torch.Tensor(testxliste).view(1, -1, 7)

testyliste = []
for k in range(7, Ntest-7):
    tab = test[k:k+7]
    newtab = [float(elmt[1])/dn for elmt in tab]
    testyliste.append(newtab)
evaluation = torch.Tensor(testyliste[:92]).view(1, -1, 7)
testy = torch.Tensor(testyliste).view(1, -1, 7)


# trainx = 1, seqlen, 1
# trainy = 1, seqlen, 1
trainds = torch.utils.data.TensorDataset(trainx, trainy)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)
testds = torch.utils.data.TensorDataset(testx, testy)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)
crit = nn.MSELoss()


class Mod(nn.Module):
    def __init__(self, nhid, N):
        super(Mod, self).__init__()
        self.rnn = nn.RNN(7, nhid, batch_first=True)
        self.mlp = nn.Linear(nhid, N)
        self.nhid = nhid

    def forward(self, x):
        # x.shape = (1,356,7) -> (N,L,Hin)
        # rnn with batch_first needs (N,L,Hin)
        y, h = self.rnn(x)
        y = torch.relu(y)
        # y.shape = (N,L,D*Hout) --> (1,356,1)
        # h.shape = (D*num_layers,N,Hout) --> (1,1,10)
        h = h.view(-1, self.nhid)  # = h.squeeze()
        y = self.mlp(h)
        return y.squeeze(0)


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
    h = 10
    nepochs = 20
    tab = range(1)
    lr = 0.005
    tab_value = []
    tab_loss = []
    tab = [21]*100
    for _ in tab:
        mod = Mod(h, 7)
        tab_train, tab_test = train(mod, nepochs, lr)
        tens = evaluation
        tab_value.append(mod(tens)[-1].item()*50)
    mean = sum(tab_value)/len(tab_value)
    print(f"max value = {max(tab_value)}")
    print(f"min value = {min(tab_value)}")
    print(f"average value = {mean}")
    print(f"écart-type = {np.std(tab_value)}")
    print("real value = 21.0")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.array(tab_value), x=np.arange(100),
                             mode='markers', name="Prédiction du RNN"))
    fig.add_trace(go.Scatter(y=np.array([mean]*100), x=np.arange(100),
                             mode='lines', name="Moyenne du RNN"))
    fig.add_trace(go.Scatter(y=np.array(tab), x=np.arange(100),
                             mode='lines', name="Valeur réelle"))
    fig.update_layout(
        title=f"Prédiction de la température du 07/04/2020 par le RNN",
        xaxis_title="Exécution",
        yaxis_title="Température (°C)"
    )
    fig.show()
