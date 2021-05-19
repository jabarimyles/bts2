import os
import pandas
import pickle
import urllib.request
from tqdm import tqdm



fp = './data/table_dict.pickle'
f = open(fp, 'rb')
td = pickle.load(f)

bat_ids = list(td['statcast']['batter'].unique())
pit_ids = list(td['statcast']['pitcher'].unique())
ids = bat_ids + pit_ids


base_url = 'https://securea.mlb.com/mlb/images/players/head_shot/{}@2x.jpg'
save_fp = './static/images/{}.jpg'
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
current_files = list(os.listdir('./static/images'))
for id in tqdm(ids):
    try:
        if '{}.jpg'.format(id) not in current_files:
            urllib.request.urlretrieve(base_url.format(int(id)), save_fp.format(id))
    except:
        print('error on id: {}'.format(id))
