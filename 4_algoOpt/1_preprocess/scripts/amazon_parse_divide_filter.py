import gzip
import json
import argparse
import math
import sys
import random
import copy
from tqdm import tqdm
from pathlib import Path
from collections import Counter, OrderedDict

######## USAGE #########
#sys.argv[1]: filename
#sys.argv[2]: min frequency (10 if not given)

##### IMPORTANT #####
##  Must user python >= 3.6
#######################

data_path = str(Path.home())+'/MERCI/data/1_raw_data/amazon/'

def parse(path):
    g = gzip.open(path, 'r')
    for l in tqdm(g, desc='Reading metadata', mininterval=1.0):
        yield eval(l)
    g.close()

split_ratio = 0.8

also_viewed_len = 0
also_buy_len = 0
item_map = dict()
max_relation_len = -1

# Read metadata file and save it to dictionary per item
for l in parse(data_path + "meta_" + sys.argv[1] + ".json.gz"):
    related_items = set()
    if 'also_view' in l:
        related_items.update(l['also_view'])
        also_viewed_len += len(l['also_view'])
    if 'also_buy' in l:
        related_items.update(l['also_buy'])
        also_buy_len += len(l['also_buy'])
    if len(related_items) > 0:
        item_map[l['asin']] = related_items
    if len(related_items) > max_relation_len:
        max_relation_len = len(related_items)
print("Average also_view+also_buy : ", (also_viewed_len+also_buy_len)/len(item_map))
print("Maximum also_view+also_buy : ", max_relation_len)

review_per_train_item = OrderedDict()
# Read review data and count each item and review
with gzip.open(data_path + sys.argv[1] + ".json.gz", "r") as review_file:
    long_item_list = list()
    for line in tqdm(review_file, desc="Reading review file", mininterval=1.0):
        itemid = (json.loads(line))["asin"]
        if itemid in item_map:
            if itemid in review_per_train_item:
                review_per_train_item[itemid] += 1
            else:
                review_per_train_item[itemid] = 1
            long_item_list += list(item_map[itemid])
    count_per_item = Counter(long_item_list)
    print("Before filtering, train+test item num : ", len(count_per_item))

filtered_items = set()
# Get set of items over frequency 10
min_frequency = 10
if len(sys.argv) == 3:
    min_frequency = int(sys.argv[2])
print("Filtering items appeared less than: ", min_frequency)
for item in count_per_item:
    if count_per_item[item] >= min_frequency:
        filtered_items.add(item)
print("After filtering, train+test item num : ", len(filtered_items))

# Delete review with zero also_view+also_buy
delete_keys = list()
for key in item_map:
    item_map[key].intersection_update(filtered_items)
    if len(item_map[key]) == 0:
        delete_keys.append(key)
for key in tqdm(delete_keys, desc="Deleting reviews with 0 items", mininterval=1.0):
    del item_map[key]
    if key in review_per_train_item:
        del review_per_train_item[key]

# Delete review item under 10 frequency
delete_keys = list()
for review in review_per_train_item:
    if review not in filtered_items:
        delete_keys.append(review)
for key in tqdm(delete_keys, desc="Deleting items under min_support", mininterval=1.0):
    del review_per_train_item[key]

# Create partition index map for weighted sampling
map_count = 0
partition_index_map = dict()
for review in review_per_train_item:
    count = review_per_train_item[review]
    for i in range(count):
        partition_index_map[map_count] = review
        map_count+=1

assert len(partition_index_map) == sum(review_per_train_item.values())

total_weighted_review_num = sum(review_per_train_item.values())
test_num = int(total_weighted_review_num*(1-split_ratio))
random.seed(0)
sample_index = random.sample(range(0, total_weighted_review_num), test_num)

# Split train/test set
review_per_test_item = OrderedDict()
for si, i in tqdm(enumerate(range(test_num)), desc="Splitting train/test datset", mininterval=1.0):
    item = partition_index_map[sample_index[si]]
    # Update train review dict
    if review_per_train_item[item] > 1:
        review_per_train_item[item] -= 1
        # Update test review dict
        if item in review_per_test_item:
            review_per_test_item[item] += 1
        else:
            review_per_test_item[item] = 1
    else:
        del review_per_train_item[item]

# Gather train set
pin_sum = 0
train_set = set()
for item in review_per_train_item:
    pin_sum += len(item_map[item])
    train_set.update(item_map[item])

# Gather test set
test_set = set()
for item in review_per_test_item:
    test_set.update(item_map[item])

only_test_set = list(test_set.difference(train_set))
random.shuffle(only_test_set)

train_test_set = list()
train_test_set += train_set
random.shuffle(train_test_set)
train_test_set += only_test_set
print("Train+Test item num : ", len(train_test_set), " / Train item num : ", len(train_set), " / Test item num : ", len(test_set), " / Only test set : ", len(only_test_set))

if len(train_test_set) == 0:
    exit(1)

# Building index map
train_test_dict = dict()
for key,val in enumerate(train_test_set):
    train_test_dict[val] = key+1

train_transaction_len = 0
test_transaction_len = 0
train_test_transactions = sum(review_per_train_item.values())+sum(review_per_test_item.values())
train_transactions = sum(review_per_train_item.values())
test_transactions = sum(review_per_test_item.values())
print("# of train+test transactions : ", train_test_transactions, "\n# of train transactions: ", train_transactions, "\n# of test transactions: ", test_transactions)

#Write to train file
with open(str(Path.home())+"/MERCI/data/4_filtered/amazon_"+sys.argv[1]+"/amazon_"+sys.argv[1]+"_train_filtered.txt", "w") as train_write:
    train_write.write("1 "+str(len(train_set)) + " " + str(len(review_per_train_item))+" "+str(pin_sum)+" 2\n")
    for review in tqdm(review_per_train_item, desc="Writing to train file", mininterval=1.0):
        train_write.write(str(review_per_train_item[review])+" ")
        train_write.write(' '.join(map(str, [train_test_dict[item] for item in item_map[review]]))+" \n")
        train_transaction_len += len(item_map[review])*review_per_train_item[review]
        
#Write to test file
with open(str(Path.home())+"/MERCI/data/4_filtered/amazon_"+sys.argv[1]+"/amazon_"+sys.argv[1]+"_test_filtered.txt", "w") as test_write:
    test_write.write("# "+str(len(train_test_set))+"\n")  
    for review in tqdm(review_per_test_item, desc="Writing to test file", mininterval=1.0):
        test_write.write(str(review_per_test_item[review]) + " ")
        test_write.write(' '.join(map(str, [train_test_dict[item] for item in item_map[review]]))+" \n")
        test_transaction_len += len(item_map[review])*review_per_test_item[review]

print("Average train+test transaction len : ", float(train_transaction_len+test_transaction_len)/train_test_transactions)
print("Average train transaction len : ", float(train_transaction_len)/train_transactions)
print("Average test transaction len : ", float(test_transaction_len)/test_transactions)