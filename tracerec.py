'''
This code trains the TraceRec model for the given dataset. 
'''

import time
from library_data import *
import library_models as lib
from library_models import *
import datetime
from evaluate_all_ import eval_one_epoch
from graph import *

now = datetime.datetime.now()
# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser('tracerec')

# select dataset and training mode
parser.add_argument('--dataset', default='wikipedia', type=str, help='Network name', choices=['wikipedia', 'mooc', 'lastfm', 'douban_movie', 'video', 'yoochoosebuy'],)
parser.add_argument('--model', default='tracerec',type=str, help="Model name")
parser.add_argument('--gpu', default=1, type=str, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions {32, 64, ''128'', 256}')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')

# general training hyper-parameters
parser.add_argument('--weight_decay', type=float, default=1e-5, help='l2 penalty')  
parser.add_argument('--learning_rate', type=float, default=1e-3, help='l2 penalty')  
parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--span_num', default=500, type=int, help='time span number')
parser.add_argument('--early_stop', type=int, default=5, help='link or link_sign or sign')
parser.add_argument('--tolerance', type=float, default=1e-3, help='tolerated marginal improvement for early stopper')

# method-related hyper-parameters

# options
parser.add_argument('--state_change', action='store_true', help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 
parser.add_argument('--online_test', action='store_true', help='Enable online test mode')

# temporal walk
parser.add_argument('--walk_type', type=str, default='before', choices=['before', 'point'], help='two type of temporal random walk type')
parser.add_argument('--num_path_u', type=int, default=16, help='number of paths(i.e., alpha1)')
parser.add_argument('--num_path_i', type=int, default=16, help='number of paths(i.e., alpha2)')
parser.add_argument('--len_path', type=int, default=2, help='length of each path(i.e., beta)')
parser.add_argument('--aggregation_method', default='lstm',type=str, help="aggregation_method (ex, GRU, lstm, concat)")
parser.add_argument('--project', action='store_true', help='Enable projection mode (default: false)')
parser.add_argument('--title', type=str, default='none', help='detail method')

args = parser.parse_args()

set_random_seed(args.seed)

args.datapath = "/home/user/KMJ/DyRec/dataset/%s/%s.csv" % (args.dataset, args.dataset)
args.user_feature_path = "/home/user/KMJ/DyRec/dataset/%s/user_feat.csv" % (args.dataset)
args.item_feature_path = "/home/user/KMJ/DyRec/dataset/%s/item_feat.csv" % (args.dataset)
log_path = './log/'
best_model_root = './best_models/'
checkpoint_root = './saved_checkpoints/'

logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys.argv, now, log_path, checkpoint_root, best_model_root)

if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, y_true] = load_network(args)

num_interactions = len(user_sequence_id)
num_users = len(user2id) + 1
num_items = len(item2id) + 1 # one extra item for "none-of-these"

# Load Feature
if args.dataset == "douban_movie":
    item_feat_flag = True
    user_feat_flag = False
    num_features = 0 
else:
    item_feat_flag = False
    user_feat_flag = False
    num_features = len(feature_sequence[0])

if args.dataset == "douban_movie":
    user_feature, item_feature = load_feature(args, item_feat_flag, user_feat_flag, item2id, user2id)
else: 
    user_feature, item_feature = None, None

if item_feat_flag == True:
    if num_items - 1 != len(item_feature):
        print("item feature error")
    else:
        num_features += len(item_feature[1])
if user_feat_flag == True:
    if num_users != len(user_feature):
        print("user feature error")
    else:
        num_features += len(user_feature[1])

true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error. 
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

item_sequence_id_new = [item_id + num_users for item_id in item_sequence_id]
total_nodes = num_users + num_items

full_adj_list = [[] for _ in range(total_nodes)] # for test set
partial_adj_list = [[] for _ in range(total_nodes)] # for train & validation set

for eidx, (src, dst, ts) in enumerate(zip(user_sequence_id, item_sequence_id_new, timestamp_sequence)):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))

for eidx, (src, dst, ts) in enumerate(zip(user_sequence_id, item_sequence_id_new, timestamp_sequence)):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
    if eidx == test_start_idx:
        break

full_ngh_finder = NeighborFinder(full_adj_list, walk_type=args.walk_type, bias=0, sample_method='multinomial')
partial_ngh_finder = NeighborFinder(partial_adj_list, walk_type=args.walk_type,bias=0, sample_method='multinomial')

# SET BATCHING TIMESPAN
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / args.span_num     

# INITIALIZE MODEL AND PARAMETERS
model = TraceRec(args, num_features, num_users, num_items).cuda()
model.update_ngh_finder(partial_ngh_finder)
weight = torch.Tensor([1,true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()
early_stopper = EarlyStopMonitor(max_round=args.early_stop , tolerance=args.tolerance ) 

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
item_embeddings_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
user_embeddings_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings 

# INITIALIZE MODEL
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# RUN MODEL

print("*** Training the TraceRec model for %d epochs ***" % args.epochs)

# variables to help using tbatch cache between epochs
is_first_epoch = True
cached_tbatches_user = {}
cached_tbatches_item = {}
cached_tbatches_interactionids = {}
cached_tbatches_feature = {}
cached_tbatches_user_timediffs = {}
cached_tbatches_item_timediffs = {}
cached_tbatches_previous_item = {}
cached_tbatches_timestamp = {}

mean_learning_time_per_epoch = []
mean_inference_time_per_epoch = []

for ep in range(args.epochs):
    # INITIALIZE EMBEDDING TRAJECTORY STORAGE
    user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
    item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

    optimizer.zero_grad()
    reinitialize_tbatches()
    total_loss, loss, total_interaction_count = 0, 0, 0

    tbatch_start_time = None
    tbatch_to_insert = -1
    tbatch_full = False

    # TRAIN TILL THE END OF TRAINING INTERACTION IDX
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    epoch_start_time = time.time()
    for j in range(train_end_idx):
        if is_first_epoch:
            # READ INTERACTION J
            userid = user_sequence_id[j]
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

            # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
            tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
            lib.tbatchid_user[userid] = tbatch_to_insert
            lib.tbatchid_item[itemid] = tbatch_to_insert

            lib.current_tbatches_user[tbatch_to_insert].append(userid)
            lib.current_tbatches_item[tbatch_to_insert].append(itemid)
            lib.current_tbatches_feature[tbatch_to_insert].append(feature)
            lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
            lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
            lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
            lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])
            lib.current_tbatches_timestamp[tbatch_to_insert].append(timestamp)

        timestamp = timestamp_sequence[j]
        if tbatch_start_time is None:
            tbatch_start_time = timestamp

        # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

            # ITERATE OVER ALL T-BATCHES
            if not is_first_epoch:
                lib.current_tbatches_user = cached_tbatches_user[timestamp]
                lib.current_tbatches_item = cached_tbatches_item[timestamp]
                lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                lib.current_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                lib.current_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                lib.current_tbatches_previous_item = cached_tbatches_previous_item[timestamp]
                lib.current_tbatches_timestamp = cached_tbatches_timestamp[timestamp]

            # with trange(len(lib.current_tbatches_user), ascii=True) as progress_bar3:
            for i in range(len(lib.current_tbatches_user)):
                total_interaction_count += len(lib.current_tbatches_interactionids[i])

                # LOAD THE CURRENT TBATCH
                if is_first_epoch:
                    lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).cuda()
                    lib.current_tbatches_item[i] = torch.LongTensor(lib.current_tbatches_item[i]).cuda()
                    lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                    lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i]).cuda()

                    lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()
                    lib.current_tbatches_item_timediffs[i] = torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()
                    lib.current_tbatches_previous_item[i] = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()
                    lib.current_tbatches_timestamp[i] = torch.Tensor(lib.current_tbatches_timestamp[i]).cuda()

                tbatch_userids = lib.current_tbatches_user[i] # Recall "lib.current_tbatches_user[i]" has unique elements
                tbatch_itemids = lib.current_tbatches_item[i] # Recall "lib.current_tbatches_item[i]" has unique elements
                tbatch_interactionids = lib.current_tbatches_interactionids[i]
                feature_tensor = Variable(lib.current_tbatches_feature[i]) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                user_timediffs_tensor = Variable(lib.current_tbatches_user_timediffs[i]).unsqueeze(1)
                item_timediffs_tensor = Variable(lib.current_tbatches_item_timediffs[i]).unsqueeze(1)
                tbatch_itemids_previous = lib.current_tbatches_previous_item[i]
                item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]
                tbatch_times = lib.current_tbatches_timestamp[i]
                # PROJECT USER EMBEDDING TO CURRENT TIME
                user_embedding_input = user_embeddings[tbatch_userids,:]
                user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, causal_embeddings=None, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[tbatch_itemids_previous,:], user_embeddings_static[tbatch_userids,:]], dim=1)

                # PREDICT NEXT ITEM EMBEDDING                            
                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                # CALCULATE PREDICTION LOSS
                item_embedding_input = item_embeddings[tbatch_itemids,:]
                loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embeddings_static[tbatch_itemids,:]], dim=1).detach())

                temporal_path_user = model.grab_subgraph(tbatch_userids, tbatch_times, args.len_path, args.num_path_u) # graph class 호출
                temporal_path_item = model.grab_subgraph(tbatch_itemids + num_users, tbatch_times, args.len_path, args.num_path_i)

                # Output: path 구조
                temporal_path_user = model.subgraph_tree2walk(tbatch_userids.cpu().numpy(), tbatch_times.cpu().numpy(), temporal_path_user)
                temporal_path_item = model.subgraph_tree2walk(tbatch_itemids.cpu().numpy() + num_users, tbatch_times.cpu().numpy(), temporal_path_item)

                u_neighbor_records, u_time_decay_weights, u_masks, u_time_feat = model.forward_msg_time_delta(temporal_path_user) 
                i_neighbor_records, i_time_decay_weights, i_masks, i_time_feat = model.forward_msg_time_delta(temporal_path_item)

                u_final_embeddings, u_valid_mask = model.get_valid_embeddings_vectorized(u_neighbor_records, u_time_feat,  u_masks, user_embeddings, item_embeddings, 'user')
                i_final_embeddings, i_valid_mask = model.get_valid_embeddings_vectorized(i_neighbor_records, i_time_feat, i_masks, item_embeddings, user_embeddings, 'item')

                u_walk_final_embeddings = model.aggregate_embeddigs(u_final_embeddings, u_valid_mask, u_time_decay_weights, \
                                                                    aggregation_method=args.aggregation_method, project=args.project)
                i_walk_final_embeddings = model.aggregate_embeddigs(i_final_embeddings, i_valid_mask, i_time_decay_weights, \
                                                                    aggregation_method=args.aggregation_method, project=args.project)


                # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                user_embedding_output = model.forward(user_embedding_input, item_embedding_input, causal_embeddings=u_walk_final_embeddings, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                item_embedding_output = model.forward(user_embedding_input, item_embedding_input, causal_embeddings=i_walk_final_embeddings, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                item_embeddings[tbatch_itemids,:] = item_embedding_output
                user_embeddings[tbatch_userids,:] = user_embedding_output  

                user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output

            # BACKPROPAGATE ERROR AFTER END OF T-BATCH
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # RESET LOSS FOR NEXT T-BATCH
            loss = 0
            item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_() 
            user_embeddings_timeseries.detach_()
            
            # REINITIALIZE
            if is_first_epoch:
                cached_tbatches_user[timestamp] = lib.current_tbatches_user
                cached_tbatches_item[timestamp] = lib.current_tbatches_item
                cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                cached_tbatches_user_timediffs[timestamp] = lib.current_tbatches_user_timediffs
                cached_tbatches_item_timediffs[timestamp] = lib.current_tbatches_item_timediffs
                cached_tbatches_previous_item[timestamp] = lib.current_tbatches_previous_item
                cached_tbatches_timestamp[timestamp] = lib.current_tbatches_timestamp

                reinitialize_tbatches()
                tbatch_to_insert = -1

    is_first_epoch = False # as first epoch ends here
    torch.cuda.synchronize()
    learning_time_per_epoch = time.time()-epoch_start_time

    # memory
    train_peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    train_current_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    if ep > 0:  
        mean_learning_time_per_epoch.append(learning_time_per_epoch)

    val_mrr, val_rec1, val_rec5, val_rec10, val_rec20, val_pre1, val_pre5, val_pre10, val_pre20, val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20, inference_time_per_epoch, val_peak_memory_mb = eval_one_epoch(args, model, optimizer, MSELoss, crossEntropyLoss, weight, 
                   user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, 
                   user_embeddings, item_embeddings, user_embeddings_static, item_embeddings_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   validation_start_idx, test_start_idx, num_users)
    

    mean_inference_time_per_epoch.append(inference_time_per_epoch)
   
    # End of epoch
    logger.info('epoch: {}'.format(ep))
    logger.info('Total loss in this epoch: {}'.format(total_loss))
    logger.info('val MRR: {:.4f}'.format(val_mrr))
    logger.info('val REC:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_rec1, val_rec5, val_rec10, val_rec20))
    logger.info('val PRE:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_pre1, val_pre5, val_pre10, val_pre20))
    logger.info('val NDCG:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20))
    logger.info('Trainig time: {} sec'.format(learning_time_per_epoch))
    logger.info('Inference time: {} sec'.format(inference_time_per_epoch))
    logger.info('Training peak memory: {:.2f} MiB (current: {:.2f} MiB)'.format(train_peak_memory_mb, train_current_memory_mb))  
    logger.info('Inference peak memory: {:.2f} MiB'.format(val_peak_memory_mb))  


    if early_stopper.early_stop_check(val_mrr):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_checkpoint_path = get_checkpoint_path(early_stopper.best_epoch)

        model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, best_checkpoint_path)
        set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx) 

        item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
        item_embeddings = item_embeddings.clone()
        item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
        item_embeddings_static = item_embeddings_static.clone()

        user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
        user_embeddings = user_embeddings.clone()
        user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
        user_embeddings_static = user_embeddings_static.clone()
        
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        model.eval()
        break
    else:
        item_embeddings_dystat = torch.cat([item_embeddings, item_embeddings_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embeddings_static], dim=1)

        save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries, get_checkpoint_path(ep))
        
        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)


# Save final model of final epoch
logger.info("\n***** Training complete. Testing final model. *****\n")
model.update_ngh_finder(full_ngh_finder)

test_mrr, test_rec1, test_rec5, test_rec10, test_rec20, test_pre1, test_pre5, test_pre10, test_pre20, test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20, inference_time, test_peak_memory_mb = eval_one_epoch(args, model, optimizer, MSELoss, crossEntropyLoss, weight, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, 
                   user_embeddings, item_embeddings, user_embeddings_static, item_embeddings_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   test_start_idx, test_end_idx, num_users)

logger.info('Best epoch: {}'.format(early_stopper.best_epoch))
logger.info('test MRR:\t{:.4f}'.format(test_mrr))
logger.info('test REC:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_rec1, test_rec5, test_rec10, test_rec20))
logger.info('test PRE:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_pre1, test_pre5, test_pre10, test_pre20))
logger.info('test NDCG:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20))
logger.info('Average training time per epoch: {}'.format(np.mean(mean_learning_time_per_epoch)))
logger.info('Average inference time per epoch: {}'.format(np.mean(mean_inference_time_per_epoch)))
logger.info('Inference time (testing): {} sec'.format(inference_time))
logger.info('Inference peak memory: {:.2f} MiB'.format(test_peak_memory_mb)) 

torch.save(model.state_dict(), best_model_path)