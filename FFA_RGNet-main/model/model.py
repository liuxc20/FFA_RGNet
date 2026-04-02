import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool
import math

class GNNModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        out = global_mean_pool(x, batch) 
        return out


class FFA_RGNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        feat_dim = 2048
        emb_dim = 32

        # multi-task head
        self.phase_head = nn.Linear(feat_dim, 3)
        self.location_head = nn.Linear(feat_dim, 7)
        self.lesion_head = nn.Linear(feat_dim, 7)

        # label embedding
        self.phase_emb = nn.Embedding(3, emb_dim)
        self.location_emb = nn.Embedding(7, emb_dim)
        self.lesion_emb = nn.Linear(7, emb_dim) 

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # GNN
        self.gnn_out_dim = 32
        self.gnn = GNNModule(in_dim=emb_dim, hidden_dim=64, out_dim=self.gnn_out_dim)
        
        self.score_layer = nn.Linear(feat_dim, 1)
        self.max_entropy = math.log(7)
        
        # diagnosis head
        self.diagnosis_head = nn.Linear(feat_dim + self.gnn_out_dim, 7)

    def forward(self, images, mask=None):
        device = images.device  
        B, T, C, H, W = images.shape
        images = images.view(B * T, C, H, W)
        feat = self.backbone(images) 

        phase_logits = self.phase_head(feat)
        location_logits = self.location_head(feat)
        lesion_logits = self.lesion_head(feat)

        phase_ids = phase_logits.argmax(dim=-1)
        location_ids = location_logits.argmax(dim=-1)
        lesion_scores = torch.sigmoid(lesion_logits)


        feat = feat.view(B, T, -1).permute(1, 0, 2)

        src_key_padding_mask = ~mask.bool() if mask is not None else None
        transformer_output = self.transformer_encoder(feat, src_key_padding_mask=src_key_padding_mask)
        transformer_output = transformer_output.permute(1, 0, 2) 
        
        scores = self.score_layer(transformer_output).squeeze(-1)  
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))  
        weights = torch.softmax(scores, dim=1)  # (B, T)
        pooled = torch.sum(transformer_output * weights.unsqueeze(-1), dim=1) 

        # bulid graph
        graph_data, subgraph_counts = self._build_graph(phase_ids, location_ids, lesion_scores, B, T, mask, weights)
        if graph_data is not None:
            gnn_out = self.gnn(graph_data.x, graph_data.edge_index, graph_data.batch)
            split_gnn_out = torch.split(gnn_out, subgraph_counts, dim=0)
            gnn_out_pooled = torch.stack([x.mean(dim=0) if len(x) > 0 else torch.zeros(self.gnn_out_dim, device=images.device) for x in split_gnn_out], dim=0)
        else:
            gnn_out_pooled = torch.zeros(B, self.gnn_out_dim, device=images.device)

        fusion_feat = torch.cat([pooled, gnn_out_pooled], dim=-1)
        diagnosis_logits = self.diagnosis_head(fusion_feat)
        diagnosis_probs = F.softmax(diagnosis_logits, dim=1)     

        # confidence
        confidence, _ = diagnosis_probs.max(dim=1)    

        # normalized_entropy
        entropy = -torch.sum(diagnosis_probs * torch.log(diagnosis_probs + 1e-8), dim=1) 
        normalized_entropy = entropy / self.max_entropy   

        phase_logits = phase_logits.view(B, T, -1)
        location_logits = location_logits.view(B, T, -1)
        lesion_logits = lesion_logits.view(B, T, -1)

        return phase_logits, location_logits, lesion_logits, diagnosis_logits, confidence, normalized_entropy

    def _build_graph(self, phase_ids, location_ids, lesion_scores, B, T, mask, weights):
        device = phase_ids.device
        data_list = []
        node_offset = 0
        phase_node_indices = []
        all_node_types = []
        all_time_idx = [] 
        all_node_values = []  
        threshold = 0.05  
        subgraph_counts = [0 for _ in range(B)]



        for b in range(B):
            for t in range(T):
                if not mask[b, t]:
                    continue
                if weights[b, t] < threshold:
                    continue 

                idx = b * T + t
                phase_node = self.phase_emb(phase_ids[idx])
                location_node = self.location_emb(location_ids[idx])
                lesion_pos = (lesion_scores[idx] > 0.5).nonzero(as_tuple=True)[0]

                if lesion_pos.numel() > 0:
                    lesion_input = torch.eye(7, device=device)[lesion_pos]
                    lesion_node_feats = self.lesion_emb(lesion_input)
                    x = torch.vstack([
                        phase_node.unsqueeze(0),
                        location_node.unsqueeze(0),
                        lesion_node_feats
                    ])
                    node_types = ['Phase', 'Location'] + ['Lesion'] * lesion_node_feats.size(0)
                    node_values = [int(phase_ids[idx].item()), int(location_ids[idx].item())] + lesion_pos.tolist()
                else:
                    x = torch.vstack([
                        phase_node.unsqueeze(0),
                        location_node.unsqueeze(0)
                    ])
                    node_types = ['Phase', 'Location']
                    node_values = [int(phase_ids[idx].item()), int(location_ids[idx].item())]

                N = x.size(0)
                edge_index = []

                # Phase <-> Location
                edge_index += [[0, 1], [1, 0]]

                # Location <-> Lesions
                for i in range(2, N):
                    edge_index += [[1, i], [i, 1]]

                # Lesion <-> Lesion
                for i in range(2, N):
                    for j in range(i + 1, N):
                        edge_index += [[i, j], [j, i]]

                edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long, device=device)

                # 全局 Phase 节点位置
                phase_node_indices.append(node_offset)
                node_offset += N

                time_idx = torch.full((N,), -1, dtype=torch.long, device=device)
                time_idx[0] = b * T + t 

                data = Data(x=x, edge_index=edge_index)
                data.node_type = node_types
                data.node_value = node_values
                data.img_index = b * T + t  

                data_list.append(data)
                subgraph_counts[b] += 1
                all_node_types.extend(node_types)
                all_node_values.extend(node_values)
                all_time_idx.append(time_idx)

        if len(data_list) == 0:
            return None

        # 跨子图 Phase-Phase 边
        global_edge_index = []
        for i in range(len(phase_node_indices) - 1):
            src = phase_node_indices[i]
            tgt = phase_node_indices[i + 1]
            global_edge_index += [[src, tgt], [tgt, src]]

        batch_data = Batch.from_data_list(data_list)        

        if len(global_edge_index) > 0:
            global_edge_index = torch.tensor(global_edge_index, dtype=torch.long, device=device).t().contiguous()
            batch_data.edge_index = torch.cat([batch_data.edge_index, global_edge_index], dim=1)
            
        batch_data.node_type = all_node_types
        batch_data.node_value = all_node_values
        batch_data.time_idx = torch.cat(all_time_idx, dim=0)    

        return batch_data, subgraph_counts
