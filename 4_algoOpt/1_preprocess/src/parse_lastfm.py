import math
import os
import sys

data_path = sys.argv[1]
path1 = 'lastfm_1'
path2 = 'lastfm_2'
dirs = [chr(65 + i) for i in range(26)]
song_similars_map = dict()
songs = set()

# Read json files
print('reading json files...')
for dir1 in dirs:
    print(dir1)
    for dir2 in dirs:
        for dir3 in dirs:
            for train_test_path in [path1, path2]:
                dirname = '{}/{}/{}/{}/{}/'.format(data_path, train_test_path, dir1, dir2, dir3)
                if not os.path.isdir(dirname):
                    continue
                for entry in os.listdir(dirname):
                    if entry.endswith('.json'):
                        entry_file = open(dirname + entry, 'r')
                        entry_dict = eval(entry_file.read())
                        entry_file.close()
                        song_similars_map[entry_dict['track_id']] = [similar[0] for similar in entry_dict['similars']]
                        songs.add(entry_dict['track_id'])
                        for song in song_similars_map[entry_dict['track_id']]:
                            songs.add(song)

print('\nremapping transactions...')
songs_list = list(songs)
random.shuffle(songs_list)
songs_index = dict()
for idx, song in enumerate(songs_list):
    songs_index[song] = idx + 1
transactions = dict()
for song in song_similars_map:
    if len(song_similars_map[song]) == 0:
        continue
    transactions[songs_index[song]] = [songs_index[similar] for similar in song_similars_map[song]]

transfile = open(os.getenv("HOME")+"/MERCI/data/2_transactions/lastfm/lastfm_transactions.txt", 'w')
for song in transactions:
    transfile.write(' '.join(map(str, transactions[song])) + '\n')
transfile.close()
