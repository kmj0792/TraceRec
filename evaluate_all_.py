from library_data import *
from library_models import *
import time

def eval_one_epoch(args, model, optimizer, MSELoss, crossEntropyLoss, weight, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, 
                   user_embeddings, item_embeddings, user_embeddings_static, item_embeddings_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   start_idx, end_idx, num_users, filter_users=None):
    # PERFORMANCE METRICS
    ranks = []
    inference_time_per_epoch = 0  

    torch.cuda.reset_peak_memory_stats()
    tbatch_start_time = None
    loss = 0
    # FORWARD PASS
    print("*** Making interaction predictions by forward pass (no t-batching) ***")
    # epoch_start_time = time.time()
    num_items = np.shape(item_embeddings)[0]
    timespan = timestamp_sequence[-1] - timestamp_sequence[0]
    tbatch_timespan = timespan / args.span_num

    for j in range(start_idx, end_idx):

        userid = int(user_sequence_id[j])
        itemid = item_sequence_id[j]
        if item_feat_flag == True and user_feat_flag == True:
            feature = user_feature[userid] + item_feature[itemid]
        elif item_feat_flag == True and user_feat_flag == False:
            feature = item_feature[itemid]
        elif item_feat_flag == False and user_feat_flag == True:
            feature = user_feature[userid]
        else:
            feature = feature_sequence[j]
        user_timediff = user_timediffs_sequence[j]
        item_timediff = item_timediffs_sequence[j]
        timestamp = timestamp_sequence[j]
        if not tbatch_start_time:
            tbatch_start_time = timestamp
        itemid_previous = user_previous_itemid_sequence[j]

        torch.cuda.synchronize()
        epoch_start_time = time.time()
        # LOAD USER AND ITEM EMBEDDING
        user_embedding_input = user_embeddings[torch.tensor([userid], dtype=torch.long, device='cuda')]
        user_embedding_static_input = user_embeddings_static[torch.tensor([userid], dtype=torch.long, device='cuda')]
        item_embedding_input = item_embeddings[torch.tensor([itemid], dtype=torch.long, device='cuda')]
        item_embedding_static_input = item_embeddings_static[torch.tensor([itemid], dtype=torch.long, device='cuda')]
        feature_tensor = torch.tensor(feature, dtype=torch.float32, device='cuda').unsqueeze(0)
        user_timediffs_tensor = torch.tensor([user_timediff], dtype=torch.float32, device='cuda').unsqueeze(0)
        item_timediffs_tensor = torch.tensor([item_timediff], dtype=torch.float32, device='cuda').unsqueeze(0)
        item_embedding_previous = item_embeddings[torch.tensor([itemid_previous], dtype=torch.long, device='cuda')]
        
        # PROJECT USER EMBEDDING
        user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, causal_embeddings=None, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.tensor([itemid_previous], dtype=torch.long, device='cuda')], user_embedding_static_input], dim=1)
        # PREDICT ITEM EMBEDDING
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # CALCULATE PREDICTION LOSS
        if args.online_test:
            loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())
        
        # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS 
        euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), torch.cat([item_embeddings, item_embeddings_static], dim=1)).squeeze(-1) 
        
        # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
        true_item_distance = euclidean_distances[itemid]
        euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
        true_item_rank = np.sum(euclidean_distances_smaller) + 1

        ranks.append(true_item_rank)
        
        temporal_path_user = model.grab_subgraph([userid], [timestamp], args.len_path, args.num_path_u)
        temporal_path_item = model.grab_subgraph([itemid + num_users], [timestamp], args.len_path, args.num_path_i)

        subgraph_src_ = model.subgraph_tree2walk(np.array([userid]), np.array([timestamp]), temporal_path_user)
        subgraph_tgt_ = model.subgraph_tree2walk(np.array([itemid + num_users]), np.array([timestamp]), temporal_path_item)

        u_neighbor_records, u_time_decay_weights, u_masks, u_time_feat = model.forward_msg_time_delta(subgraph_src_) 
        i_neighbor_records, i_time_decay_weights, i_masks, i_time_feat = model.forward_msg_time_delta(subgraph_tgt_) 
        
        u_final_embeddings, u_valid_mask = model.get_valid_embeddings_vectorized(u_neighbor_records, u_time_feat, u_masks, user_embeddings, item_embeddings, 'user')
        i_final_embeddings, i_valid_mask = model.get_valid_embeddings_vectorized(i_neighbor_records, i_time_feat, i_masks, item_embeddings, user_embeddings, 'item')

        u_walk_final_embeddings = model.aggregate_embeddigs(u_final_embeddings, u_valid_mask, u_time_decay_weights, \
                                                            aggregation_method=args.aggregation_method, project=args.project)
        i_walk_final_embeddings = model.aggregate_embeddigs(i_final_embeddings, i_valid_mask, i_time_decay_weights, \
                                                            aggregation_method=args.aggregation_method, project=args.project)

        # UPDATE USER AND ITEM EMBEDDING
        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, causal_embeddings=u_walk_final_embeddings, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update') 
        item_embedding_output = model.forward(user_embedding_input, item_embedding_input, causal_embeddings=i_walk_final_embeddings, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update') 

        # SAVE EMBEDDINGS
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0) 
        user_embeddings[userid,:] = user_embedding_output.squeeze(0) 
        user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
        item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0)
        
        # UPDATE THE MODEL IN REAL-TIME USING ERRORS MADE IN THE PAST PREDICTION
        if args.online_test:
            if timestamp - tbatch_start_time > tbatch_timespan:
                tbatch_start_time = timestamp
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # RESET LOSS FOR NEXT T-BATCH
                loss = 0      
                item_embeddings.detach_()
                user_embeddings.detach_()
                item_embeddings_timeseries.detach_() 
                user_embeddings_timeseries.detach_()
        else:
            item_embeddings.detach_()
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_()
            user_embeddings_timeseries.detach_()
        
        torch.cuda.synchronize()
        inference_time_per_epoch += (time.time()-epoch_start_time)
    # CALCULATE THE PERFORMANCE METRICS
    
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    mrr = np.mean([1.0 / r for r in ranks])
    
    #recall_at_k
    rec1 = sum(np.array(ranks) <= 1)*1.0 / len(ranks)
    rec5 = sum(np.array(ranks) <= 5)*1.0 / len(ranks)
    rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)
    rec20 = sum(np.array(ranks) <= 20)*1.0 / len(ranks)
    
    # precision_at_k
    pre1 = sum(np.array(ranks) <= 1) / (len(ranks) * 1)
    pre5 = sum(np.array(ranks) <= 5) / (len(ranks) * 5)
    pre10 = sum(np.array(ranks) <= 10) / (len(ranks) * 10)
    pre20 = sum(np.array(ranks) <= 20) / (len(ranks) * 20)

    ndcg1 = ndcg_at_k(ranks, 1)
    ndcg5 = ndcg_at_k(ranks, 5)
    ndcg10 = ndcg_at_k(ranks, 10)
    ndcg20 = ndcg_at_k(ranks, 20)

    return mrr, rec1, rec5, rec10, rec20, pre1, pre5, pre10, pre20, ndcg1, ndcg5, ndcg10, ndcg20, inference_time_per_epoch, peak_memory_mb


def ndcg_at_k(ranks, k):
    ndcgs = []
    for r in ranks:
        if r <= k:
            dcg = 1.0 / math.log2(r + 1)
            idcg = 1.0 
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)
    return np.mean(ndcgs)