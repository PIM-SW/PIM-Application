import math
import random
import sys
from tqdm import tqdm

'''
train_test_division.py

common parser that divides given raw transactions file into train/train(thresholded)/test sets

usage:
    python3 train_test_division.py transactions.txt threshold

arguments:
    transactions.txt: a text file that contains all transactions of a dataset
                      no header, 1 transaction per line, each item in a transaction denoted by an integer separated by whitespace
    threshold (optional): filters out train dataset with given threshold
                          if not given, no filtering is executed
'''

def parse_transactions(transfile_name, train_ratio=0.8, threshold=0.0, verbose_interval_ratio = 0.1):
    '''
        transfile_name: string that identifies a .txt file that contains transactions;
                        the file must contain 1 transaction per line, 
                        each item denoted as an integer separated by single space
    '''
    transfile = open(transfile_name, 'r')
    transactions = dict()
    user_idx = 1
    for transaction in tqdm(transfile, desc='Reading transactions data', mininterval=1.0):
        transactions[user_idx] = list(set(map(int, transaction.split(' ')))) # Remove duplicate items within transaction
        user_idx += 1
    transfile.close()

    num_transactions = len(transactions)
    verbose_interval = int(num_transactions * verbose_interval_ratio)

    # Randomly divide into train/test sets
    users = list(transactions.keys())
    random.seed(1)
    random.shuffle(users)
    num_train = int(num_transactions * train_ratio)
    transactions_train = dict()
    transactions_test = dict()
    count = 0
    for user in tqdm(users, desc='Dividing into train/test sets', mininterval=1.0):
        items = transactions[user]
        if count < num_train:
            transactions_train[user] = items
        else:
            transactions_test[user] = items
        count += 1

    # Collect thresholded appearances, if threshold given
    if (threshold > 0):
        items_train_frequency = dict()
        num_transactions_train = len(transactions_train)
        verbose_interval = int(num_transactions_train * verbose_interval_ratio)
        for user in tqdm(transactions_train, desc='Collecting frequent items', mininterval=1.0):
            for item in transactions_train[user]:
                items_train_frequency.setdefault(item, 0)
                items_train_frequency[item] += 1
        frequent_items = set()
        least_appearance = int(num_transactions_train * threshold)
        for item in items_train_frequency:
            if items_train_frequency[item] >= least_appearance:
                frequent_items.add(item)

        # Attain transactions with frequently appearing items only
        transactions_top = dict()
        count = 0
        for user in tqdm(transactions_train, desc='Building new train set', mininterval=1.0):
            for item in transactions_train[user]:
                if item in frequent_items:
                    transactions_top.setdefault(user, []).append(item)
        transactions_train = transactions_top
        transactions = {**transactions_train, **transactions_test}

    # Set weights
    weights = dict()
    for user in transactions_train:
        weights[user] = 1
    for user in transactions_test:
        weights[user] = 1

    # Remove duplicates
    if True: # Turn off for datasets w/o duplicates
        print('Removing duplicates...')
        for transactions in [transactions_train, transactions_test]:
            num_transactions = len(transactions)
            verbose_interval = int(num_transactions * verbose_interval_ratio)
            transactions_string = dict()
            transactions_string_rev = dict()
            count = 0
            for user in tqdm(transactions, desc='String hasing', mininterval=1.0):
                curr_tr = sorted(transactions[user])
                hashed_transaction = str(curr_tr)
                transactions_string[user] = hashed_transaction
                transactions_string_rev.setdefault(hashed_transaction, []).append(user)

            hts = list(transactions_string_rev.keys())
            for ht in tqdm(hts, desc='Identifying duplicates', mininterval=1.0):
                users = transactions_string_rev[ht]
                if len(users) > 1:
                    weights[users[0]] = len(users)
                    for i in range(1, len(users)):
                        transactions.pop(users[i], None) 
        transactions = {**transactions_train, **transactions_test}

    # Collect stats
    all_items = None
    train_items = None
    train_stats = None # needed for writing train data
    if threshold > 0:
        statfile = open(transfile_name.split('/')[-1][:-17] + '_stats_'+str(threshold)+'.txt', 'w')
    else:
        statfile = open(transfile_name.split('/')[-1][:-17] + '_stats.txt', 'w')
    statnames = ['all', 'train', 'test']
    for idx, transactions in enumerate([transactions, transactions_train, transactions_test]):
        avg_trans = 0
        min_trans = math.inf
        max_trans = -1
        user_cnt = len(transactions)
        items = list()
        item_cnt = 0
        interaction_cnt = 0
        verbose_interval = int(user_cnt * verbose_interval_ratio)
        for user in tqdm(transactions, desc='Collecting stats for {}'.format(statnames[idx]), mininterval=1.0):
            trans_size = len(transactions[user])
            items.extend(transactions[user])
            avg_trans += trans_size
            if min_trans > trans_size:
                min_trans = trans_size
            if max_trans < trans_size:
                max_trans = trans_size
            interaction_cnt += trans_size
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
    statfile.close()

    items_index = dict()
    idx = 1
    for item in tqdm(train_items, desc='Reindexing train items', mininterval=1.0):
        items_index[item] = idx
        idx += 1
    for item in tqdm(all_items, desc='Reindexing remaining items', mininterval=1.0):
        if item not in train_items:
            items_index[item] = idx
            idx += 1

    if threshold > 0:
        transfile = open(transfile_name.split('/')[-1][:-17] + '_train_'+str(threshold)+'_weighted.txt', 'w')
    else:
        transfile = open(transfile_name.split('/')[-1][:-17] + '_train.txt', 'w')
    transfile.write(' '.join(map(str, train_stats)) + '\n')
    for user in tqdm(transactions_train, desc='Writing train transaction data', mininterval=1.0):
        transfile.write(str(weights[user]) + ' ' + ' '.join(str(items_index[item]) for item in transactions_train[user]) + '\n')
    transfile.close()

    if threshold > 0:
        transfile = open(transfile_name.split('/')[-1][:-17] + '_test_'+str(threshold)+'_weighted.txt', 'w')
    else:
        transfile = open(transfile_name.split('/')[-1][:-17] + '_test.txt', 'w')
    transfile.write('# {}\n'.format(idx - 1))
    for user in tqdm(transactions_test, desc='Writing test transaction data', mininterval=1.0):
        transfile.write(str(weights[user]) + ' ' + ' '.join(str(items_index[item]) for item in transactions_test[user]) + '\n')
    transfile.close()


if __name__ == '__main__':
    transactions_filename = sys.argv[1]
    if len(sys.argv) > 2:
        parse_transactions(transactions_filename, threshold=eval(sys.argv[2]))
    else:
        parse_transactions(transactions_filename)
