from argparse import ArgumentParser, Namespace
import torch
from tqdm import tqdm
from collections import Counter
import pickle
import pandas as pd
import os
import json
import re
import numpy as np
from utils import Vocab
import random

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\[\]\/#!$%\^\*;:{}=\_`~()@<>]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def build_dictionary(args):
    data_dict = torch.load(os.path.join(args.dataset_dir, args.dataset, 'sets/train.pt'))
    song_list = []
    token_list = []
    artist_list = []
    for instance in data_dict:
        song_list.extend(instance['songs'])
    for instance in data_dict:
        artist_list.extend(instance['artist_ids'])
    for instance in data_dict:
        token_list.extend(instance['nrm_plylst_title'].split())
    s_counter = Counter(song_list)
    t_counter = Counter(token_list)
    a_counter = Counter(artist_list)
    s_vocab = Vocab(list_of_tokens=list(s_counter.keys()))
    t_vocab = Vocab(list_of_tokens=list(t_counter.keys()))
    a_vocab = Vocab(list_of_tokens=list(a_counter.keys()))
    with open(os.path.join(args.dataset_dir, args.dataset, 'tokenizer/title_split', args.dataset + "_vocab.pkl"), mode="wb") as io:
        pickle.dump(t_vocab, io)
    with open(os.path.join(args.dataset_dir, args.dataset, 'tokenizer/track', args.dataset + "_vocab.pkl"), mode="wb") as io:
        pickle.dump(s_vocab, io)
    with open(os.path.join(args.dataset_dir, args.dataset, 'tokenizer/artist', args.dataset + "_vocab.pkl"), mode="wb") as io:
        pickle.dump(a_vocab, io)
    return list(s_counter.keys()), list(a_counter.keys()), list(t_counter.keys())

def unk_token(val_test_data, track_vocab_list, artist_vocab_list, title_vocab_list):
    for i in tqdm(range(len(val_test_data))):
        item = val_test_data[i]
        val_test_data[i]['original_title'] = item['nrm_plylst_title']
        song_list = []
        artist_list = []
        token_list = []
        for song in item['songs']:
            if song not in track_vocab_list:
                song_list.append('<unk>')
            else:
                song_list.append(song)
        val_test_data[i]['songs'] = song_list
        
        for artist in item['artist_ids']:
            if artist not in artist_vocab_list:
                artist_list.append('<unk>')
            else:
                artist_list.append(artist)
        val_test_data[i]['artist_ids'] = artist_list

        for token in item['nrm_plylst_title'].split(" "):
            if token not in title_vocab_list:
                token_list.append('<unk>')
            else:
                token_list.append(token)
        val_test_data[i]['nrm_plylst_title'] = (" ").join(token_list)
    return val_test_data

def filter_title(dataset, title, songs, min_avg_chr_len = 2, min_title_len = 3, min_track_num = 2):
    n_tracks = len(songs)
    tokens = title.split(' ')
    mean_token_len = np.array([len(i) for i in tokens]).mean() if len(tokens) else 0
    tag_list = get_tag_list(dataset)
    if mean_token_len >= min_avg_chr_len and len(tokens) >= min_title_len and n_tracks >= min_track_num:
        for tag in tag_list:
            if tag in title:
                return True
        return False
    else:
        return False

def get_artists_set(tracks, melon_songs):
    artist_ids = []
    for track in tracks:
        artist_id = melon_songs[melon_songs['id']==track]['artist_id_basket'].values
        artist_ids += artist_id[0]

    return list(set(artist_ids))

def get_artists_names(tracks, melon_songs):
    artist_names = []
    for track in tracks:
        artist_name = melon_songs[melon_songs['id']==track]['artist_name_basket'].values
        artist_names += artist_name[0]

    return list(set(artist_names))

def get_tag_list(dataset):
    if dataset == 'melon':
        with open('./tag/korean_tag_list.txt', 'rb') as file:
            tag_list = pickle.load(file)
            
    elif dataset == 'million':
        with open('./tag/english_tag_list.txt', 'rb') as file:
            tag_list = pickle.load(file)
    
    return tag_list

def load_and_filter(args, min_avg_chr_len = 2, min_title_len = 3, min_track_num = 2):
    tqdm.pandas()
    if args.dataset == 'melon':
        dataset = 'melon'
        melon_train = pd.read_json(os.path.join(args.dataset_dir, args.dataset, "data", "train.json"))
        melon_val = pd.read_json(os.path.join(args.dataset_dir, args.dataset, "data", "val.json"))
        melon_test = pd.read_json(os.path.join(args.dataset_dir, args.dataset, "data", "test.json"))
        melon_songs = pd.read_json(os.path.join(args.dataset_dir, args.dataset,"data", "song_meta.json"))
        
        melon_playlist = pd.concat([melon_train, melon_val, melon_test], axis=0)
        melon_playlist = melon_playlist[melon_playlist['plylst_title'].map(lambda r: len(r.split(' '))>0 if not r=='' else False)]
        melon_playlist = melon_playlist[melon_playlist['songs'].map(lambda r: len(r)>0)]

        playlist = melon_playlist[['id', 'plylst_title', 'songs', 'tags', 'updt_date']]
        playlist = playlist.rename(columns={'id': 'pid'})
        playlist['nrm_plylst_title'] = playlist['plylst_title'].progress_map(normalize_name)
        filtered_playlist = playlist[playlist.progress_apply(lambda row: filter_title(dataset = args.dataset, title = row['nrm_plylst_title'], songs = row['songs']), axis=1)]
        filtered_playlist['artist_ids']= filtered_playlist['songs'].progress_map(lambda tracks: get_artists_set(tracks, melon_songs))
        filtered_playlist['artist_names']= filtered_playlist['songs'].progress_map(lambda tracks: get_artists_names(tracks, melon_songs))
    
    elif args.dataset == 'million':
        fname = os.listdir(args.dataset_dir + "/million" + "/data")
        
        dfs = []
        for file in tqdm(fname):
            if file.startswith("mpd.slice.") and file.endswith(".json"):
                file = json.load(open(os.path.join(args.dataset_dir, "million", "data", file),'r'))
                df = pd.DataFrame.from_dict(file['playlists'])
                df = df[df['name'].map(lambda r: len(r.split(' '))>0 if not r=='' else False)]
                df['songs'] = df['tracks'].map(lambda tracks: [track['track_uri'] for track in tracks if len(tracks)>0])
                df['artist_ids'] = df['tracks'].map(lambda tracks: list(set([track['artist_uri'] for track in tracks])))
                df['artist_names'] = df['tracks'].map(lambda tracks: list(set([track['artist_name'] for track in tracks])))
                df = df.rename(columns={'name': 'plylst_title'})
                df = df.rename(columns={'modified_at': 'updt_date'})
                dfs.append(df[['pid', 'updt_date', 'plylst_title', 'songs', 'artist_ids', 'artist_names']])
            
            
        playlist = pd.concat(dfs)
        
        playlist['nrm_plylst_title'] = playlist['plylst_title'].progress_map(normalize_name)
        filtered_playlist = playlist[playlist.progress_apply(lambda row: filter_title(dataset = args.dataset, title = row['nrm_plylst_title'], songs = row['songs']), axis=1)]
        
    filtered_playlist_dict = filtered_playlist.to_dict(orient='records')
    torch.save(filtered_playlist_dict, os.path.join(args.dataset_dir, args.dataset, 'sets/filtered.pt'))
    
    
def chronological_split(args):
    all = torch.load(os.path.join(args.dataset_dir, args.dataset, 'sets/filtered.pt'))
    train_data = [] 
    val_test_data = []
    
    if args.dataset == 'melon':
        for item in all:
            if item['updt_date'][:4] == '2020':
                val_test_data.append(item)
            else:
                train_data.append(item)
    if args.dataset == 'million':
        start_date = 1506816000
        for item in all:
            if item['updt_date'] >= start_date:
                val_test_data.append(item)
            else:
                train_data.append(item)
    torch.save(train_data, os.path.join(args.dataset_dir, args.dataset, 'sets/train.pt'))
    
    song_list, artist_list, title_vocab_list = build_dictionary(args)
    unk_token(val_test_data, song_list, artist_list, title_vocab_list)
    random.shuffle(val_test_data)
    length = int(len(val_test_data) / 2)
    val_data = val_test_data[:length]
    test_data = val_test_data[length:]
    torch.save(val_data, os.path.join(args.dataset_dir, args.dataset, 'sets/val.pt'))
    torch.save(test_data, os.path.join(args.dataset_dir, args.dataset, 'sets/test.pt'))

def artist_frequency_split(args):
    train_data = torch.load(os.path.join(args.dataset_dir, args.dataset, 'sets/train.pt'))
    test_data = torch.load(os.path.join(args.dataset_dir, args.dataset, 'sets/test.pt'))
    
    artist_list = []
    for data in train_data:
        artist_list.extend(data['artist_ids'])
    artist_counts = Counter(artist_list)
    
    artist_frequency_dictionary = {}
    for data in test_data:
        total_counts = 0
        for artist in data['artist_ids']:
            total_counts += artist_counts[artist]
        average = total_counts / len(data['artist_ids'])
        dictionary_element = {data['pid'] : average}
        artist_frequency_dictionary.update(dictionary_element)
    artist_frequency_counter = Counter(artist_frequency_dictionary)
    length = int(len(artist_frequency_counter) / 4)
    most_common = artist_frequency_counter.most_common()[:length]
    least_common = artist_frequency_counter.most_common()[-1 * length:]
    most_common_list = [item[0] for item in most_common]
    least_common_list = [item[0] for item in least_common]
    most_common_data = []
    least_common_data = []
    for data in test_data:
        if data['pid'] in most_common_list:
            most_common_data.append(data)
        elif data['pid'] in least_common_list:
            least_common_data.append(data)
    torch.save(most_common_data, os.path.join(args.dataset_dir, args.dataset, 'sets/highest_fa.pt'))
    torch.save(least_common_data, os.path.join(args.dataset_dir, args.dataset, 'sets/lowest_fa.pt'))


def track_frequency_split(args):
    train_data = torch.load(os.path.join(args.dataset_dir, args.dataset, 'sets/train.pt'))
    test_data = torch.load(os.path.join(args.dataset_dir, args.dataset, 'sets/test.pt'))
    
    track_list = []
    for data in train_data:
        track_list.extend(data['songs'])
    track_counts = Counter(track_list)
    
    track_frequency_dictionary = {}
    for data in test_data:
        total_counts = 0
        for track in data['songs']:
            total_counts += track_counts[track]
        average = total_counts / len(data['songs'])
        dictionary_element = {data['pid'] : average}
        track_frequency_dictionary.update(dictionary_element)
    track_frequency_counter = Counter(track_frequency_dictionary)
    length = int(len(track_frequency_counter) / 4)
    most_common = track_frequency_counter.most_common()[:length]
    least_common = track_frequency_counter.most_common()[-1 * length:]
    most_common_list = [item[0] for item in most_common]
    least_common_list = [item[0] for item in least_common]
    most_common_data = []
    least_common_data = []
    for data in test_data:
        if data['pid'] in most_common_list:
            most_common_data.append(data)
        elif data['pid'] in least_common_list:
            least_common_data.append(data)
    torch.save(most_common_data, os.path.join(args.dataset_dir, args.dataset, 'sets/highest_ft.pt'))
    torch.save(least_common_data, os.path.join(args.dataset_dir, args.dataset, 'sets/lowest_ft.pt'))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="million", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    args = parser.parse_args()

    load_and_filter(args)
    chronological_split(args)
    artist_frequency_split(args)
    track_frequency_split(args)
