import sys
import math
import os
from tqdm import tqdm

'''
usage: python3 filter_by_occurrence.py [dataset_name]

filters train/test dataset by its items' occurrence in train data set
'''

# Hyper-parameters
filter_freq = 10 # only items occuring above this threshold will remain

# Attain datasets
base_dir = os.getenv("HOME")+"/MERCI/data"
train = base_dir+"/3_train_test/"+sys.argv[1]+"/"+sys.argv[1]+"_train.txt"
test = base_dir+"/3_train_test/"+sys.argv[1]+"/"+sys.argv[1]+"_test.txt"

# Attain weights, transactions, items
train_weights = []
train_transactions = []
train_items_freq = dict()
test_weights = []
test_transactions = []
test_items_freq = dict()
with open(train, 'r') as trainfile:
    trainfile.readline() # Skip header
    for line in tqdm(trainfile, desc='Attaining train data'):
        data = list(map(int, line.split()))
        train_weights.append(data[0])
        train_transactions.append(data[1:])
        for item in data[1:]:
            train_items_freq[item] = 0
with open(test, 'r') as testfile:
    testfile.readline() # Skip header
    for line in tqdm(testfile, desc='Attaining test data'):
        data = list(map(int, line.split()))
        test_weights.append(data[0])
        test_transactions.append(data[1:])
        for item in data[1:]:
            test_items_freq[item] = 0

# Attain frequencies
for transaction in tqdm(train_transactions, desc='Attaining frequencies'):
    for item in transaction:
        train_items_freq[item] += 1
        if item in test_items_freq:
            test_items_freq[item] += 1

# Filter
train_transactions_filtered = []
test_transactions_filtered = []
for transaction in tqdm(train_transactions, desc='Filtering train data'):
    transaction_filtered = []
    for item in transaction:
        if train_items_freq[item] >= filter_freq:
            transaction_filtered.append(item)
    train_transactions_filtered.append(transaction_filtered)
for transaction in tqdm(test_transactions, desc='Filtering test data'):
    transaction_filtered = []
    for item in transaction:
        if test_items_freq[item] >= filter_freq:
            transaction_filtered.append(item)
    test_transactions_filtered.append(transaction_filtered)

# Remove duplicate and remap weights
train_items = set()
all_items = set()
train_items_len = 0
train_weights_fixed = []
test_weights_fixed = []
train_hashed = dict()
test_hashed = dict()
train_transactions = []
test_transactions = []
for i in tqdm(range(len(train_transactions_filtered)), desc='Removing duplicates in train set'):
    transaction = train_transactions_filtered[i]
    curr_tr = sorted(transaction)
    transaction_hashed = str(curr_tr)
    for item in transaction:
        train_items.add(item)
        all_items.add(item)
    if len(transaction) == 0:
        continue
    if transaction_hashed in train_hashed:
        train_weights_fixed[train_hashed[transaction_hashed]] += train_weights[i]
        continue
    train_transactions.append(transaction)
    train_weights_fixed.append(train_weights[i])
    train_hashed[transaction_hashed] = len(train_weights_fixed) - 1
    train_items_len += len(transaction)
for i in tqdm(range(len(test_transactions_filtered)), desc='Removing duplicates in test set'):
    transaction = test_transactions_filtered[i]
    curr_tr = sorted(transaction)
    transaction_hashed = str(curr_tr)
    for item in transaction:
        all_items.add(item)
    if len(transaction) == 0:
        continue
    if transaction_hashed in test_hashed:
        test_weights_fixed[test_hashed[transaction_hashed]] += test_weights[i]
        continue
    test_transactions.append(transaction)
    test_weights_fixed.append(test_weights[i])
    test_hashed[transaction_hashed] = len(test_weights_fixed) - 1
train_weights = train_weights_fixed
test_weights = test_weights_fixed

# Remap
items_map = dict()
idx = 1
items_not_in_train = 0
for item in tqdm(train_items, desc='Remapping train items'):
    items_map[item] = idx
    idx += 1
for item in tqdm(all_items, desc='Remapping rest...'):
    if item not in train_items:
        items_map[item] = idx
        idx += 1
        items_not_in_train += 1

# Write back
with open(base_dir+"/4_filtered/"+sys.argv[1]+"/"+sys.argv[1]+'_train_filtered.txt', 'w') as trainfile:
    trainfile.write('{} {} {} {} {}\n'.format(
        1, len(train_items), len(train_transactions), train_items_len, 2))
    for i in tqdm(range(len(train_transactions)), desc='Writing train data'):
        trainfile.write(str(train_weights[i]) + ' ' + ' '.join(
            str(items_map[item]) for item in train_transactions[i]) + '\n')
with open(base_dir+"/4_filtered/"+sys.argv[1]+"/"+sys.argv[1]+'_test_filtered.txt', 'w') as testfile:
    testfile.write('# {}\n'.format(idx - 1))
    for i in tqdm(range(len(test_transactions)), desc='Writing test data'):
        testfile.write(str(test_weights[i]) + ' ' + ' '.join(
            str(items_map[item]) for item in test_transactions[i]) + '\n')

with open(base_dir+"/4_filtered/"+sys.argv[1]+"/"+sys.argv[1]+'_stat_filtered.txt', 'w') as statfile:
    statnames = ['train', 'test']   
    all_avg_trans = 0
    all_min_trans = math.inf
    all_max_trans = -1
    all_usr_cnt = 0
    all_interaction_cnt = 0
    for idx, transactions in enumerate([train_transactions, test_transactions]):
        avg_trans = 0
        min_trans = math.inf
        max_trans = -1
        user_cnt = len(transactions)
        all_usr_cnt += len(transactions)
        items = list()
        item_cnt = 0
        interaction_cnt = 0
        for user in tqdm(transactions, desc='Collecting stats for {}'.format(statnames[idx]), mininterval=1.0):
            trans_size = len(user)
            items.extend(user)
            avg_trans += trans_size
            all_avg_trans += trans_size
            if min_trans > trans_size:
                min_trans = trans_size
            if all_min_trans > trans_size:
                all_min_trans = trans_size
            if max_trans < trans_size:
                max_trans = trans_size
            if all_max_trans < trans_size:
                all_max_trans = trans_size
            interaction_cnt += trans_size
            all_interaction_cnt += trans_size
        avg_trans /= user_cnt
        items = set(items)
        item_cnt = len(items)
        if idx == 0:
            all_items = items
        elif idx == 1:
            train_items = items
            train_stats = [1, item_cnt, user_cnt, interaction_cnt, 2]

        # Write to files
        print('Writing stat data...')
        statfile.write("____" + statnames[idx] + "____\n")
        statfile.write(
'''
Average Transaction Size : {}
Minimum Transaction Size : {}
Maximum Transaction Size : {}
Total Number of Users : {}
Total Number of Items : {}
Total Number of Interactions : {}\n\n
'''.format(avg_trans, min_trans, max_trans, user_cnt, item_cnt, interaction_cnt)
        )
    statfile.write("____All____\n")
    statfile.write(
'''
Average Transaction Size : {}
Minimum Transaction Size : {}
Maximum Transaction Size : {}
Total Number of Users : {}
Total Number of Items : {}
Total Number of Interactions : {}
# of items only in test set : {}\n\n
        '''.format(all_avg_trans/all_usr_cnt, all_min_trans, all_max_trans, all_usr_cnt, len(all_items), all_interaction_cnt, items_not_in_train)
    )
    statfile.close()