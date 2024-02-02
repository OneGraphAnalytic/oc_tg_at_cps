# This code achieves a performance of around 96.60%. However, it is not
# directly comparable to the results reported by the TGN paper since a
# slightly different evaluation setup is used here.
# In particular, predictions in the same batch are made in parallel, i.e.
# predictions for interactions later in the batch have no access to any
# information whatsoever about previous interactions in the same batch.
# On the contrary, when sampling node neighborhoods for interactions later in
# the batch, the TGN paper code has access to previous interactions in the
# batch.
# While both approaches are correct, together with the authors of the paper we
# decided to present this version here as it is more realsitic and a better
# test bed for future methods.

import os.path as osp

import torch
import numpy as np 
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from torch_geometric.data import TemporalData

# from torch_geometric.datasets import JODIEDataset
from datasets.edge_iiot import EdgeIIoTset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from sklearn.metrics import classification_report, f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
# dataset = JODIEDataset(path, name='wikipedia')
path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data/edge-iiot_submission', 'nids')
dataset = EdgeIIoTset(path, name='DNN-EdgeIIoT-dataset')
data = dataset[0]
neg_data = dataset[1]
TRAIN_DATA_SIZE = 100000 # 750000 without features works well 
BATCH_SIZE= 2000
EVALUATION_STRATEGY = 'no' # 'attack_knowledge' or 'predicted_knowledge' or 'blind' or 'no'
THRESHOLD_EVAL = 0.5
NEIGHBOR_SIZE= 50
LEARNING_RATE = 0.001
# data.msg = torch.ones(data.msg.shape[0], 31) # dummy msg 

print(f"Train data size: {TRAIN_DATA_SIZE}, Batch size: {BATCH_SIZE}, Evaluation strategy: {EVALUATION_STRATEGY}, Threshold: {THRESHOLD_EVAL}, Neighbor size: {NEIGHBOR_SIZE}, Learning rate: {LEARNING_RATE}")
train_data = data[data.t < TRAIN_DATA_SIZE]
neg_train_data = neg_data[neg_data.t < TRAIN_DATA_SIZE]




combined_data_len = train_data.src.shape[0] + neg_train_data.src.shape[0]
combined_data_msg_features = train_data.msg.shape[1]
combined_src = torch.zeros(combined_data_len, dtype=torch.long)
combined_dst = torch.zeros(combined_data_len, dtype=torch.long)
combined_t = torch.zeros(combined_data_len, dtype=torch.long)
combined_msg = torch.zeros((combined_data_len, combined_data_msg_features), dtype=torch.float)
combined_y = torch.zeros(combined_data_len, dtype=torch.long)

for i in range(train_data.src.shape[0]):
    combined_src[i*2] = train_data.src[i]
    combined_src[i*2+1] = neg_train_data.src[i]
    combined_dst[i*2] = train_data.dst[i]
    combined_dst[i*2+1] = neg_train_data.dst[i]
    combined_t[i*2] = train_data.t[i]
    combined_t[i*2+1] = neg_train_data.t[i]
    combined_msg[i*2] = train_data.msg[i]
    combined_msg[i*2+1] = neg_train_data.msg[i]
    combined_y[i*2] = train_data.y[i]
    combined_y[i*2+1] = neg_train_data.y[i]

combined_train = TemporalData(
    src=combined_src,  
    dst=combined_dst,
    t=combined_t,
    msg=combined_msg,
    y=combined_y
)

eval_data = data[data.t >= TRAIN_DATA_SIZE]
train_loader = TemporalDataLoader(
    combined_train,
    batch_size=BATCH_SIZE,
    neg_sampling_ratio=0.0,
)
print("Train data size: ", combined_train.num_events)
print("Eval data size: ", eval_data.num_events)
print("Combined train: ", combined_train)
print("eval_data: ", eval_data) 
test_loader = TemporalDataLoader(
    eval_data,
    batch_size=BATCH_SIZE,
    neg_sampling_ratio=0.0,
)
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NEIGHBOR_SIZE, device=device)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


memory_dim = time_dim = embedding_dim = 100

memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        # pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        # neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        # loss = criterion(pos_out, torch.ones_like(pos_out))
        # loss += criterion(neg_out, torch.zeros_like(neg_out))

        out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        loss = criterion(out, batch.y.unsqueeze(-1).float())

        update_batch = batch[batch.y == 0]
        #update_batch = batch
        update_batch.src, update_batch.dst, update_batch.t, update_batch.msg, update_batch.y = batch.src[batch.y == 0], batch.dst[batch.y == 0], batch.t[batch.y == 0], batch.msg[batch.y == 0], batch.y[batch.y == 0]
            
        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(update_batch.src, update_batch.dst, update_batch.t, update_batch.msg)
        neighbor_loader.insert(update_batch.src, update_batch.dst)

        # memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        # neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / combined_train.num_events


@torch.no_grad()
def test(loader):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    y_preds = []
    y_trues = []
    for batch in loader:
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        
        out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        y_pred = out.sigmoid()
        y_true = batch.y

        y_preds.append(y_pred)
        y_trues.append(y_true)
        # pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        # neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        # y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        # y_true = torch.cat(
        #     [torch.ones(pos_out.size(0)),
        #      torch.zeros(neg_out.size(0))], dim=0)

        # aps.append(average_precision_score(y_true, y_pred))
        # aucs.append(roc_auc_score(y_true, y_pred))

        # memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        # neighbor_loader.insert(batch.src, batch.dst)

        if EVALUATION_STRATEGY == 'predicted_knowledge':
            t = torch.Tensor([THRESHOLD_EVAL])  # threshold
            y_pred_tmp = y_pred.squeeze()            
            mask = (y_pred_tmp > t)
            batch.src = batch.src[mask]
            batch.dst = batch.dst[mask]
            batch.t = batch.t[mask]
            batch.msg = batch.msg[mask]
            if batch.src.shape[0] != 0:
                memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
                neighbor_loader.insert(batch.src, batch.dst)
        elif EVALUATION_STRATEGY == 'attack_knowledge':
            batch.y = batch.y.numpy()
            mask = (batch.y == 0)
            batch.src = batch.src[mask]
            batch.dst = batch.dst[mask]
            batch.t = batch.t[mask]
            batch.msg = batch.msg[mask]
            if batch.src.shape[0] != 0:
                memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
                neighbor_loader.insert(batch.src, batch.dst)
        elif EVALUATION_STRATEGY == 'blind':
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)
        elif EVALUATION_STRATEGY == 'no':
            pass
        else:
            raise Exception('Unknown evaluation strategy')
    return y_preds, y_trues

def evaluation():
    import pandas as pd 
    print('Evaluation')
    df = pd.DataFrame({'y_preds': y_preds_rounded.flatten().astype(int)})
    df['y_trues'] = y_trues
    df['tps'] = np.where((df['y_preds'] == 1) & (df['y_trues'] == 1), 1, 0)
    df['fps'] = np.where((df['y_preds'] == 1) & (df['y_trues'] == 0), 1, 0)
    df['tns'] = np.where((df['y_preds'] == 0) & (df['y_trues'] == 0), 1, 0)
    df['fns'] = np.where((df['y_preds'] == 0) & (df['y_trues'] == 1), 1, 0)
    
    # df.to_csv(f'results/evaluation{REPORT_FILE}.csv', index=False)
    
    df_edge_iiot = pd.read_csv('data/edge-iiot_submission/nids/dnn-edgeiiot-dataset/raw/DNN-EdgeIIoT-dataset.csv')
    df_edge_iiot = df_edge_iiot[df_edge_iiot['Attack_type'] != 'DDoS_UDP']
    df_edge_iiot = df_edge_iiot[df_edge_iiot['Attack_type'] != 'MITM']
    df_edge_iiot.reset_index(drop=True, inplace=True)
    df_edge_iiot = df_edge_iiot[df_edge_iiot.index >= TRAIN_DATA_SIZE]
    df_type = df_edge_iiot[["Attack_type", "Attack_label"]].reset_index(drop=True)
    df_merged = pd.merge(df_type, df, left_index=True, right_index=True, how="inner")
    grouped_stats = df_merged.groupby('Attack_type').agg({
        'tps': 'sum',
        'fps': 'sum',
        'tns': 'sum',
        'fns': 'sum'
        }).reset_index()
    
    # Display the grouped statistics
    print(grouped_stats)

for epoch in range(1, 51):
    print(f"Update Strategy: {EVALUATION_STRATEGY}")
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    y_preds, y_trues = test(test_loader)
    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    print(f"atk f1: {f1_score(y_trues, y_preds.round())}")

    y_preds_rounded = np.where(y_preds > THRESHOLD_EVAL, 1, 0)
    print(f"Classification Report: \n {classification_report(y_trues, y_preds_rounded)}")

evaluation()