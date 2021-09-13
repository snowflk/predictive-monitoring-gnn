import torch as T
import torch.nn.functional as F
from model import PMGCN
from torch_geometric.data import DataLoader
from dataset import BPIC12Dataset
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 32
train_dataset = BPIC12Dataset('./data', train_mode=True)
val_dataset = BPIC12Dataset('./data', train_mode=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = T.device('cuda' if T.cuda.is_available() else 'cpu')
model = PMGCN(
    type_in_channels=6,
    attr_in_channels=8,
    emb_dim=32,
    hidden_dim=32).to(device)

optimizer = T.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)

criterion = T.nn.CrossEntropyLoss()


def mean(data):
    return np.mean(np.array(data))


# batch = next(iter(train_loader))
max_epoch = 300
epoch_train_loss = []
epoch_val_loss = []
epoch_train_acc = []
epoch_val_acc = []
for epoch in range(max_epoch):
    print(f"Epoch {epoch+1}/{max_epoch}")
    model.train()
    train_losses = []
    train_acc = []
    for batch_idx, batch in enumerate(train_loader):
        # print(f"Training {batch_idx}/{len(train_loader)}")
        optimizer.zero_grad()
        type_nodes, attr_nodes, edge_index = batch.type_nodes.float(), batch.attr_nodes.float(), batch.edge_index
        n_type_nodes, n_attr_nodes, global_features, batch_info = batch.n_type_nodes, batch.n_attr_nodes, batch.global_features, batch.batch

        y_truth_emb = model.emb_y(batch.y.float())
        y_truth = T.argmax(batch.y.float(), dim=1)
        y_pred_probs, y_pred_emb = model(type_nodes, attr_nodes, edge_index, n_type_nodes, n_attr_nodes,
                                         global_features.float(),
                                         batch_info)
        y_pred = T.argmax(y_pred_probs, dim=1)

        loss = criterion(y_pred_probs, y_truth)  # - F.cosine_similarity(y_truth_emb, y_pred_emb).sum()

        acc = T.sum(y_pred == y_truth) / BATCH_SIZE
        train_losses.append(loss.data)
        train_acc.append(acc.data)
        loss.backward()
        optimizer.step()

    model.eval()
    val_losses = []
    val_acc = []
    for batch_idx, batch in enumerate(val_loader):
        # print(f"Validating {batch_idx}/{len(train_loader)}")
        type_nodes, attr_nodes, edge_index = batch.type_nodes.float(), batch.attr_nodes.float(), batch.edge_index
        n_type_nodes, n_attr_nodes, global_features, batch_info = batch.n_type_nodes, batch.n_attr_nodes, batch.global_features, batch.batch
        y_truth_emb = model.emb_y(batch.y.float())
        y_truth = T.argmax(batch.y.float(), dim=1)
        y_pred_probs, y_pred_emb = model(type_nodes, attr_nodes, edge_index, n_type_nodes, n_attr_nodes,
                                         global_features.float(),
                                         batch_info)
        y_pred = T.argmax(y_pred_probs, dim=1)

        loss = criterion(y_pred_probs, y_truth)  # - F.cosine_similarity(y_truth_emb, y_pred_emb).sum()
        acc = T.sum(y_pred == y_truth) / BATCH_SIZE
        val_losses.append(loss.data)
        val_acc.append(acc.data)
    print(f"Train Loss: {mean(train_losses)} | Train Acc: {mean(train_acc)}")
    print(f"Val Loss: {mean(val_losses)} | Val Acc: {mean(val_acc)}")
    epoch_train_acc.append(mean(train_acc))
    epoch_train_loss.append(mean(train_losses))
    epoch_val_acc.append(mean(val_acc))
    epoch_val_loss.append(mean(val_losses))

T.save(model, 'saved_model.pt')
T.save({'train_acc': epoch_train_acc,
        'train_loss': epoch_train_loss,
        'val_acc': epoch_val_acc,
        'val_loss': epoch_val_loss}, 'history.pt')
