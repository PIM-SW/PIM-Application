import math
import os
import random
import sys
from tqdm import tqdm

data_path = sys.argv[1]
datafile = open(data_path+"ca-coauthors-dblp.mtx", 'r')
datafile.readline()
datafile.readline()

transactions = dict()
products = set()

for line in tqdm(datafile, desc='Reading from data file'):
    components = line.split(' ')
    author1 = int(components[0].strip())
    author2 = int(components[1].strip())
    transactions.setdefault(author1, []).append(author2)
    products.add(author2)
    transactions.setdefault(author2, []).append(author1)
    products.add(author1)

print('Number of Products: ', len(products))

# Remap indexing
print('\nreindexing...')
products = list(products)
#shuffle
print(products[0:10])
random.shuffle(products)
print(products[0:10])

products_mapping = dict()
sot = 0
for idx, product in tqdm(enumerate(products), desc='Collecting indices', total=len(products)):
    products_mapping[product] = idx + 1
    
for customer in tqdm(transactions, desc='Reindexing'):
    transactions[customer] = [products_mapping[product] for product in transactions[customer]]
    sot += len(transactions[customer])
# Save
transfile = open(os.getenv("HOME")+"/MERCI/data/2_transactions/dblp/dblp_transactions.txt", 'w')
for customer in tqdm(transactions, desc='Writing transactions'):
    transfile.write(' '.join(map(str, transactions[customer])) + '\n')
transfile.close()

print('Number of Products: ', len(products))
print('Number of Transactions: ', len(transactions))
print('SOT: ', sot)
print('Average pooling size: ', sot/len(transactions))