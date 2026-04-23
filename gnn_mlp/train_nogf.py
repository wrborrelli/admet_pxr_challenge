import numpy as np
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from model import MoleculeGNN
from utils import *
import pandas as pd
from sklearn.metrics import r2_score

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

stdout=open('lc.out','w')

dataset = MoleculeDataset(np.stack(train.SMILES), np.stack(train.pEC50))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

model = MoleculeGNN(node_dim=6, edge_dim=6, hidden_dim=32, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

stdout.write('epoch #     train loss      train R2      test R2\n')
stdout.flush()

num_epochs = 200
for epoch in range(num_epochs):

    # -------------------
    # TRAIN
    # -------------------
    model.train()

    train_preds = []
    train_targets = []
    train_loss_sum = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        pred, emb = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.edge_attr
        )

        loss = loss_fn(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

        train_preds.append(pred.detach().cpu())
        train_targets.append(batch.y.view(-1).cpu())

    train_preds = torch.cat(train_preds)
    train_targets = torch.cat(train_targets)

    train_mse = torch.mean((train_preds - train_targets) ** 2).item()
    train_r2 = r2_score(train_targets.numpy(), train_preds.numpy())

    # -------------------
    # TEST
    # -------------------
    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            pred, _ = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.edge_attr
            )

            test_preds.append(pred.cpu())
            test_targets.append(batch.y.view(-1).cpu())

    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)

    test_mse = torch.mean((test_preds - test_targets) ** 2).item()
    test_r2 = r2_score(test_targets.numpy(), test_preds.numpy())

    stdout.write(' '+str(epoch)+'   '+str(train_loss_sum / len(train_loader))+'  '+str(train_r2)+'   '+str(test_r2)+'\n')
    stdout.flush()
#    print(f"""
#Epoch {epoch:03d}
#-----------------------------
#Train Loss : {train_loss_sum / len(train_loader):.6f}
#Train MSE  : {train_mse:.6f}
#Train R²   : {train_r2:.4f}
#Test  MSE  : {test_mse:.6f}
#Test  R²   : {test_r2:.4f}
#""")
stdout.close()
torch.save(model, 'gnn_embed_66.pt')

