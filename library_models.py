from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
from itertools import chain
from tqdm.std import tqdm, trange 
import csv
import json

PATH = "./"

total_reinitialization_count = 0

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


# THE TraceRec MODULE
class TraceRec(nn.Module):
    def __init__(self, args, num_features, num_users, num_items):
        super(TraceRec,self).__init__()

        print("*** Initializing the TraceRec model ***")
        self.modelname = args.model
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items

        print("Initializing user and item embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1 + num_features

        print("Initializing user and item RNNs")
        self.item_rnn = nn.RNNCell(self.embedding_dim, self.embedding_dim)
        self.user_rnn = nn.RNNCell(self.embedding_dim, self.embedding_dim)
        
        print("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)

        self.interaction = nn.Linear(self.embedding_dim + self.embedding_dim + num_features, self.embedding_dim, bias=False)
        self.time_encoder2 = TimeEncode(expand_dim=self.embedding_dim, time_encoder_type="nonlearn")
        self.item_projection_layer = NormalLinear(1, self.embedding_dim) #mj
        self.ngh_finder = None
        self.lstm_encoder = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.embedding_dim, batch_first=True, bidirectional=False).cuda()
        self.concat_encoder = nn.Linear(3*self.embedding_dim, self.embedding_dim, bias=True)
        # GRU  
        self.gru_encoder = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.1 if 2 > 1 else 0
        )
        self.path_transformer = PathTransformerEncoder(
            d_node=self.embedding_dim,  
            d_model=self.embedding_dim,                    
            nhead=4,
            nlayers=1,
            dropout=0.1
        )
        print("Initializing re-scaling layers")
        self.LeakyReLU = nn.LeakyReLU()
        self.ReScale_Inter_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Inter_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)
        self.ReScale_Neigh_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Neigh_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)

        print("*** TraceRec initialization complete ***\n\n")
    
    def forward(self, user_embeddings, item_embeddings, causal_embeddings=None, timediffs=None, features=None, select=None):
        if select == 'item_update':
            time_features = self.time_encoder2(timediffs)
            input1 = torch.cat([user_embeddings, time_features, features], dim=1)
            input1 = self.interaction(input1)
            
            inter_embeddings = self.re_scaling(input1, 'Inter')
            causal_embeddings = self.re_scaling(causal_embeddings, 'Causal')
            inputs = inter_embeddings + causal_embeddings
            item_embedding_output = self.item_rnn(inputs, item_embeddings)
            return item_embedding_output #F.normalize(item_embedding_output)

        elif select == 'user_update':
            time_features = self.time_encoder2(timediffs)
            input2 = torch.cat([item_embeddings, time_features, features], dim=1)
            input2 = self.interaction(input2)

            inter_embeddings = self.re_scaling(input2, 'Inter')
            causal_embeddings = self.re_scaling(causal_embeddings, 'Causal')
            inputs = inter_embeddings + causal_embeddings
            user_embedding_output = self.user_rnn(inputs, user_embeddings)
            return user_embedding_output #F.normalize(user_embedding_output)

        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            return user_projected_embedding

    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out

    def re_scaling(self, embeddings, target=None):
        if embeddings is None:
            print('No embeddings to re-scale')
            return embeddings

        # Re-scaling Network
        if target == 'Inter':
            Inter_first = self.LeakyReLU(self.ReScale_Inter_first(embeddings))
            Inter_factor = self.LeakyReLU(self.ReScale_Inter_factor(Inter_first))
            return Inter_factor*embeddings#, Inter_factor
        if target == 'Causal':
            Neigh_first = self.LeakyReLU(self.ReScale_Neigh_first(embeddings))
            Neigh_factor = self.LeakyReLU(self.ReScale_Neigh_factor(Neigh_first))
            return Neigh_factor*embeddings#, Neigh_factor
        
    # Temporal Walks 
    def update_ngh_finder(self, ngh_finder): 
        self.ngh_finder = ngh_finder

    def grab_subgraph(self, src_idx_l, cut_time_l, num_layers, num_neighbors):
        subgraph = self.ngh_finder.find_k_hop(num_layers, src_idx_l, cut_time_l, num_neighbors)
        return subgraph
    
    def subgraph_tree2walk(self, src_idx_l, cut_time_l, subgraph_src):
        node_records, eidx_records, t_records = subgraph_src
        node_records_tmp = [np.expand_dims(src_idx_l, 1)] + node_records
        eidx_records_tmp = [np.full_like(node_records_tmp[0], -1)] + eidx_records
        t_records_tmp = [np.expand_dims(cut_time_l, 1)] + t_records

        new_node_records = self.subgraph_tree2walk_one_component(node_records_tmp)
        new_eidx_records = self.subgraph_tree2walk_one_component(eidx_records_tmp)
        new_t_records = self.subgraph_tree2walk_one_component(t_records_tmp)

        return new_node_records , new_eidx_records, new_t_records
    
    def subgraph_tree2walk_one_component(self, record_list):
        batch, n_walks, walk_len, dtype = record_list[0].shape[0], record_list[-1].shape[-1], len(record_list), record_list[0].dtype
        record_matrix = np.empty((batch, n_walks, walk_len), dtype=dtype)
        for hop_idx, hop_record in enumerate(record_list):
            assert(n_walks % hop_record.shape[-1] == 0)
            record_matrix[:, :, hop_idx] = np.repeat(hop_record, repeats=n_walks // hop_record.shape[-1], axis=1)
        return record_matrix
    
    def forward_msg_time_delta(self, subgraph_src_):
        node_records, edge_records, time_records = subgraph_src_  
        masks = self.get_mask(node_records)
        
        t_target = time_records[:, :, 0]
        t_bridge = time_records[:, :, 1]
        t_source = time_records[:, :, 2]

        valid_mask1 = (t_bridge >= 0) 
        valid_mask2 = (t_source >= 0) 
        delta_t1 = t_target - t_bridge
        delta_t2 = t_target - t_source

        large_value = 0
        delta_t1_masked = np.where(valid_mask1, delta_t1, large_value)
        delta_t2_masked = np.where(valid_mask2, delta_t2, large_value)

        time_features = None

        return node_records, [delta_t1_masked, delta_t2_masked], masks, time_features
    
    def get_valid_embeddings_vectorized(self, node_records, time_feat, masks, src_embeddings, tgt_embeddings, source):
        batch_size, n_walk, len_walk = node_records.shape
        device = tgt_embeddings.device
        node_records = torch.from_numpy(node_records).to(device)

        src_size = src_embeddings.shape[0]
        tgt_size = tgt_embeddings.shape[0] 
        embed_dim = src_embeddings.shape[1]

        position_indices = torch.arange(len_walk, device=device).view(1, 1, -1).expand(batch_size, n_walk, -1)
        
        valid_mask = position_indices < masks.unsqueeze(-1)  # (batch, n_walk, len_walk)
        is_even = (position_indices % 2 == 0)
        is_odd = ~is_even

        masked_node_records = node_records.clone()

        if source == 'user': 
            # even=user, odd=item
            invalid_even = ~valid_mask & is_even
            invalid_odd = ~valid_mask & is_odd

            masked_node_records[invalid_even] = 0
            masked_node_records[invalid_odd] = src_size
            
            even_ids = masked_node_records[is_even]
            odd_ids = masked_node_records[is_odd] - src_size

            even_embs = src_embeddings[even_ids]
            odd_embs = tgt_embeddings[odd_ids]
        elif source == 'item':
            # even=item, odd=user  
            invalid_even = ~valid_mask & is_even
            invalid_odd = ~valid_mask & is_odd
            
            masked_node_records[invalid_even] = tgt_size
            masked_node_records[invalid_odd] = 0
            
            even_ids = masked_node_records[is_even] - tgt_size
            odd_ids = masked_node_records[is_odd]
            
            even_embs = src_embeddings[even_ids]
            odd_embs = tgt_embeddings[odd_ids]
        
        final_embeddings = torch.empty(batch_size, n_walk, len_walk, embed_dim, device=device)
        final_embeddings[is_even] = even_embs
        final_embeddings[is_odd] = odd_embs
        
        return final_embeddings, valid_mask
    
    def context_convert_item(self, embeddings, timediffs):
        original_shape = timediffs.shape
        timediffs_flat = (-timediffs).flatten().unsqueeze(-1)
        projection_weights = self.item_projection_layer(timediffs_flat) # share
        projection_weights = projection_weights.view(original_shape[0], original_shape[1], -1)  # [batch, path, embedding_dim]

        new_embeddings = embeddings * (1 + projection_weights)
        return new_embeddings
    
    def aggregate_embeddigs(self, path_embeddings, valid_mask, weight, aggregation_method, project=True, exclude_padding=False):
        if project:
            path_embeddings = self._apply_positional_time_projections(path_embeddings, weight)
        if aggregation_method == 'lstm':
            return self._aggregate_lstm_include_padding(path_embeddings, valid_mask, weight)
        elif aggregation_method == 'concat':
            return self.aggregate_path_embeddings_concat(path_embeddings)
        elif aggregation_method == 'GRU':
            return self._aggregate_GRU_include_padding(path_embeddings, valid_mask)
        elif aggregation_method == 'transformer':
            return self.path_transformer(path_embeddings, weight, valid_mask)

    
    def _apply_positional_time_projections(self, path_embeddings, timediffs):
        timediffs1 = torch.tensor(timediffs[0], dtype=torch.float32).cuda()
        timediffs2 = torch.tensor(timediffs[1], dtype=torch.float32).cuda()
        
        target_embeddings = path_embeddings[:, :, 0, :]  # (batch, num_paths, embed_dim)
        bridge_embeddings = path_embeddings[:, :, 1, :]  
        source_embeddings = path_embeddings[:, :, 2, :]  
        
        # projection 
        projected_bridge = self.context_convert_item(bridge_embeddings, timediffs1) # B → projected B  
        projected_source = self.context_convert_item(source_embeddings, timediffs2) # B → projected B  

        # (batch, num_paths, 3, embed_dim)
        projected_embeddings = torch.stack([
            target_embeddings,   
            projected_bridge,     
            projected_source 
        ], dim=2)
        
        return projected_embeddings
    
    def _aggregate_lstm_include_padding(self, path_embeddings, valid_mask, weights=None):
        batch_size, n_walk, len_walk, embed_dim = path_embeddings.shape
        device = path_embeddings.device
        
        reversed_embeddings = torch.flip(path_embeddings, dims=[2])
        
        reversed_embeddings_flat = reversed_embeddings.view(-1, len_walk, embed_dim)
        
        lstm_out, (hidden, _) = self.lstm_encoder(reversed_embeddings_flat)
        
        embed_dim_half = int(embed_dim/2)
        final_embeddings = hidden[-1].view(batch_size, n_walk, embed_dim)

        final_embeddings = final_embeddings.mean(dim=1)

        return final_embeddings
    
    def aggregate_path_embeddings_concat(self, path_embeddings): # concat
        batch_size, n_walk, len_walk, embed_dim = path_embeddings.shape
        path_concat_embeddings = path_embeddings.view(batch_size, n_walk, len_walk * embed_dim)
        concat_embeddings = path_concat_embeddings.mean(dim=1)
        final_embeddings = self.concat_encoder(concat_embeddings)
        return final_embeddings
    
    def _aggregate_GRU_include_padding(self, path_embeddings, valid_mask):
        batch_size, n_walk, len_walk, embed_dim = path_embeddings.shape
        device = path_embeddings.device
        
        reversed_embeddings = torch.flip(path_embeddings, dims=[2])

        reversed_embeddings_flat = reversed_embeddings.view(-1, len_walk, embed_dim)
        
        gru_out, hidden = self.gru_encoder(reversed_embeddings_flat)
                
        final_embeddings = hidden[-1].view(batch_size, n_walk, embed_dim)

        final_embeddings = final_embeddings.mean(dim=1)
        return final_embeddings
    
    def get_mask(self, node_records):
        device = next(self.parameters()).device
        node_records_th = torch.from_numpy(node_records).long().to(device)
        masks = (node_records_th != -1).sum(dim=-1).long()  
        return masks
    
# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1


# CALCULATE LOSS FOR THE PREDICTED USER STATE 
def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDCIT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    y = Variable(torch.LongTensor(y_true).cuda()[tbatch_interactionids])
    
    loss = loss_function(prob, y)

    return loss


# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings, train_end_idx, user_embeddings_time_series=None, item_embeddings_time_series=None, path=PATH):
    print("*** Saving embeddings and model ***")
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx
            }

    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(state, path, pickle_protocol=4)
    print("*** Saved embeddings and model to file: %s ***\n\n" % path)


# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, args, path):
    checkpoint = torch.load(path, weights_only=False)
    print("Loading saved embeddings and model: %s" % path)

    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cuda())
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).cuda())
    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cuda())
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).cuda())
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return [model, optimizer, user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, train_end_idx]


# SET USER AND ITEM EMBEDDINGS TO THE END OF THE TRAINING PERIOD 
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()

class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round
    
class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, time_encoder_type, factor=5):
        super(TimeEncode, self).__init__()

        self.time_dim = expand_dim
        self.alpha = math.sqrt(self.time_dim)
        self.beta = math.sqrt(self.time_dim)
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
        self.parameter_requires_grad = time_encoder_type
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / self.alpha ** np.linspace(0, self.beta-1, self.time_dim))).float())

        self.w = nn.Linear(1, self.time_dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / self.alpha ** np.linspace(0, self.beta-1, self.time_dim))).float().reshape(self.time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.time_dim))

        if not self.parameter_requires_grad == 'nonlearn': 
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False


    def forward(self, ts):
        if self.parameter_requires_grad == 'learn':
            map_ts = ts * self.basis_freq
            map_ts += self.phase

            output = torch.cos(map_ts)

        elif self.parameter_requires_grad == 'nonlearn':
            ts = ts#.unsqueeze(dim=2)
            output = torch.cos(self.w(ts))
        
        return output 

class PathTransformerEncoder(nn.Module):
    def __init__(self,
                 d_node: int,            
                 d_model: int = 128,     
                 nhead: int = 4,
                 nlayers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        in_dim = d_node 
        self.proj = nn.Linear(in_dim, d_model)
        self.role_emb = nn.Embedding(3, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,        # (B, L, D)
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.out_norm = nn.LayerNorm(d_model)
    
    def forward(self,
                node_emb3: torch.Tensor,  # (B,3,d_node)  [SRC,BRG,TGT]
                weights: torch.Tensor,              
                token_valid_mask: torch.Tensor = None # (B,3) bool, False=pad
                ) -> torch.Tensor:
        B, num_paths, L, d_node = node_emb3.shape

        assert L == 3, "."

        node_emb_flat = node_emb3.view(B * num_paths, L, d_node)
        mask_flat = token_valid_mask.view(B * num_paths, L)  # (B*num_paths, 3)

        roles3 = torch.tensor([0,1,2], device=node_emb3.device).view(1,3).repeat(B* num_paths,1)
        
        h = self.proj(node_emb_flat) + self.role_emb(roles3)  # (B*num_paths, 3, d_model)
        src_key_padding_mask = ~mask_flat.bool()  # (B*num_paths, 3), True=mask

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B*num_paths, 3, d_model)
        h = self.out_norm(h)

        path_emb_flat  = h[:, 2, :]  # (B,d_model)
        d_model = path_emb_flat.shape[-1]
        path_embeddings = path_emb_flat.view(B, num_paths, d_model)  # (B, num_paths, d_model)

        final_embeddings = path_embeddings.mean(dim=1)
        return final_embeddings