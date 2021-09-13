import torch as T
import torch.nn.functional as F
from model import PMGCN
from torch_geometric.data import DataLoader
from dataset import BPIC12Dataset
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 64
train_dataset = BPIC12Dataset('./data', train_mode=True)
val_dataset = BPIC12Dataset('./data', train_mode=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print("Using device:", device)
model = PMGCN(
    type_in_channels=8,
    attr_in_channels=8,
    emb_dim=32,
    hidden_dim=32).to(device)

optimizer = T.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

criterion = T.nn.CrossEntropyLoss()
time_criterion = T.nn.L1Loss()


def mean(data):
    return np.mean(np.array(data))


# batch = next(iter(train_loader))
max_epoch = 300


class Metrics:
    def __init__(self, name):
        self.batch = []
        self.batch_history = []
        self.epoch = []
        self.name = name

    def batch_end(self, data):
        self.batch.append(data)
        self.batch_history.append(data)

    def epoch_end(self):
        self.epoch.append(self.current_epoch)
        self.batch.clear()

    @property
    def current_epoch(self):
        return mean(self.batch)

    @property
    def current_batch(self):
        return self.batch_history[-1]

    def __repr__(self):
        return f"{self.name}: {self.current_epoch}"

    def current_s(self):
        return f"{self.name}: {self.current_batch}"


train_loss = Metrics('Train Loss')
val_loss = Metrics('Val Loss')
train_mae = Metrics('Train MAE')
val_mae = Metrics('Val MAE')
train_acc = Metrics('Train Acc')
val_acc = Metrics('Val Acc')

for epoch in range(max_epoch):
    print(f"Epoch {epoch+1}/{max_epoch}==============")
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        # print(f"Training {batch_idx}/{len(train_loader)}")
        optimizer.zero_grad()
        type_nodes, attr_nodes, edge_index = batch.type_nodes.float(), batch.attr_nodes.float(), batch.edge_index
        n_type_nodes, n_attr_nodes, global_features, batch_info = batch.n_type_nodes, batch.n_attr_nodes, batch.global_features, batch.batch

        y_truth_time = batch.y_time.float()
        y_truth_emb = model.emb_y(batch.y.float())
        y_truth = T.argmax(batch.y.float(), dim=1)
        y_pred_probs, y_pred_emb, y_pred_time = model(type_nodes, attr_nodes, edge_index, n_type_nodes, n_attr_nodes,
                                                      global_features.float(),
                                                      batch_info)
        y_pred = T.argmax(y_pred_probs, dim=1)

        time_mae = time_criterion(y_pred_time.squeeze(), y_truth_time)
        loss = criterion(y_pred_probs, y_truth) - F.cosine_similarity(y_truth_emb, y_pred_emb).mean() + time_mae

        acc = T.sum(y_pred == y_truth) / BATCH_SIZE

        train_loss.batch_end(loss.cpu().data)
        train_acc.batch_end(acc.cpu().data)
        train_mae.batch_end(time_mae.cpu().data)
        print(
            f"[{batch_idx+1}/{len(train_loader)}] {train_loss.current_s()} | {train_acc.current_s()} | {train_mae.current_s()}")
        loss.backward()
        optimizer.step()

    model.eval()

    for batch_idx, batch in enumerate(val_loader):
        batch = batch.to(device)

        # print(f"Validating {batch_idx}/{len(train_loader)}")
        type_nodes, attr_nodes, edge_index = batch.type_nodes.float(), batch.attr_nodes.float(), batch.edge_index
        n_type_nodes, n_attr_nodes, global_features, batch_info = batch.n_type_nodes, batch.n_attr_nodes, batch.global_features, batch.batch

        y_truth_time = batch.y_time.float()
        y_truth_emb = model.emb_y(batch.y.float())
        y_truth = T.argmax(batch.y.float(), dim=1)
        y_pred_probs, y_pred_emb, y_pred_time = model(type_nodes, attr_nodes, edge_index, n_type_nodes, n_attr_nodes,
                                                      global_features.float(),
                                                      batch_info)
        y_pred = T.argmax(y_pred_probs, dim=1)

        time_mae = time_criterion(y_pred_time.squeeze(), y_truth_time)
        loss = criterion(y_pred_probs, y_truth) - F.cosine_similarity(y_truth_emb, y_pred_emb).mean() + time_mae

        acc = T.sum(y_pred == y_truth) / BATCH_SIZE
        val_loss.batch_end(loss.cpu().data)
        val_acc.batch_end(acc.cpu().data)
        val_mae.batch_end(time_mae.cpu().data)
        print(
            f"[{batch_idx+1}/{len(train_loader)}] {val_loss.current_s()} | {val_acc.current_s()} | {val_mae.current_s()}")
    print("===========================================")
    print(f"{train_loss} | {train_acc} | {train_mae}")
    print(f"{val_loss} | {val_acc} | {val_mae}")

    train_loss.epoch_end()
    val_loss.epoch_end()
    train_acc.epoch_end()
    val_acc.epoch_end()
    train_mae.epoch_end()
    val_mae.epoch_end()

name = "dim32"
T.save(model, f'saved_model_{name}.pt')
T.save({'epoch_train_acc': train_acc.epoch,
        'epoch_train_loss': train_loss.epoch,
        'epoch_val_acc': val_acc.epoch,
        'epoch_val_loss': val_loss.epoch,
        'train_acc': train_acc.batch_history,
        'train_loss': train_loss.batch_history,
        'val_acc': val_acc.batch_history,
        'val_loss': val_loss.batch_history}, f'history_{name}.pt')
