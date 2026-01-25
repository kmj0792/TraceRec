from __future__ import division
import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import os, re
import argparse
from sklearn.preprocessing import scale
import torch
import logging

# LOAD THE DATASET
def load_network(args, time_scaling=True):
    dataset = args.dataset
    datapath = args.datapath

    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []

    print("\n\n**** Loading %s network from file: %s ****" % (dataset, datapath))
    f = open(datapath,"r")
    if dataset != 'douban_movie':
        f.readline()
    for cnt, l in enumerate(f):
        if dataset == 'douban_movie': # FORMAT: user, item, rating, timestamp
            ls = l.strip().split("\t")
            user_sequence.append(ls[0])
            item_sequence.append(ls[1])
            if start_timestamp is None:
                start_timestamp = float(ls[3])
            timestamp_sequence.append(float(ls[3]) - start_timestamp)

        else: # FORMAT: user, item, timestamp, state label, feature list 
            ls = l.strip().split(",")
            user_sequence.append(ls[0])
            item_sequence.append(ls[1])
            if start_timestamp is None:
                start_timestamp = float(ls[2])
            timestamp_sequence.append(float(ls[2]) - start_timestamp) 
            # y_true_labels.append(int(ls[3])) # label = 1 at state change, 0 otherwise
            feature_sequence.append(list(map(float,ls[4:])))
    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formating item sequence")
    nodeid = 1
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formating user sequence")
    nodeid = 1
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        feature_sequence, \
        y_true_labels]


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_feature(args, item_feat_flag, user_feat_flag, item2id, user2id):
    if (item_feat_flag == True) and (user_feat_flag == True):       
        user_features = load_user_feat(args.user_feature_path, user2id)
        item_features = load_item_feat(args.item_feature_path, item2id)
    elif (item_feat_flag == True) and (user_feat_flag == False):
        user_features = None
        item_features = load_item_feat(args.item_feature_path, item2id)
        
    return user_features, item_features

def load_user_feat(user_feature_path, user2id):
    user_features = {}
    with open(user_feature_path, "r") as f:
        for user_id, feat in enumerate(f):
            user_features[user2id[str(user_id)]] = list(map(int, feat.strip().split()))
    return user_features

def load_item_feat(item_feature_path, item2id):
    item_features = {}
    with open(item_feature_path, "r") as f:
        for item_id, feat in enumerate(f):
            item_features[item2id[str(item_id)]] = list(map(int, feat.strip().split()))
    return item_features


def str2bool(value):
    """Converts string to boolean"""
    if value.lower() in ('yes', 'true', 't', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise ValueError("Boolean value expected")
    

def set_up_logger(args, sys_argv, now, log_path, checkpoint_root, best_model_root):
    runtime_id = '{}-{}-{}-{}-{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute,  now.second, args.dataset)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    directory = log_path + '{}/'.format(args.dataset) 
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_path = log_path + '{}/{}.log'.format(args.dataset, runtime_id) 
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)

    # checkpoint_root = './saved_checkpoints/'
    checkpoint_dir = checkpoint_root + runtime_id + '/'
    # best_model_root = './best_models/'
    best_model_dir = best_model_root + runtime_id + '/'
    if not os.path.exists(checkpoint_root):
        os.mkdir(checkpoint_root)
        logger.info('Create directory {}'.format(checkpoint_root))
    if not os.path.exists(best_model_root):
        os.mkdir(best_model_root)
        logger.info('Create directory'.format(best_model_root))
    os.mkdir(checkpoint_dir)
    os.mkdir(best_model_dir)
    logger.info('Create checkpoint directory {}'.format(checkpoint_dir))
    logger.info('Create best model directory {}'.format(best_model_dir))

    get_checkpoint_path = lambda epoch: (checkpoint_dir + 'checkpoint-epoch-{}.pth'.format(epoch))
    best_model_path = best_model_dir + 'best-model.pth'

    return logger, get_checkpoint_path, best_model_path


def get_gpu_memory_usage():
    """GPU 메모리 사용량 반환 (MB 단위)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # 현재 할당
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # 예약됨
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)   # 피크
        return allocated, reserved, peak
    return 0, 0, 0

def reset_gpu_memory_stats():
    """GPU 메모리 통계 리셋"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()