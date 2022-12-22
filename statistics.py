import torch
import os
from collections import Counter
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import matplotlib.pyplot as plt

    

def unk_proportion(val_data, test_data):
    val_test = val_data + test_data
    total_artist_length =  0
    total_track_length = 0
    unk_artist_length = 0
    unk_track_length = 0
    for item in val_test:
        artists = item['artist_ids']
        tracks = item['songs']
        total_artist_length += len(artists)
        total_track_length += len(tracks)
        artist_counter = Counter(artists)
        track_counter = Counter(tracks)
        unk_artist_length += artist_counter['<unk>']
        unk_track_length += track_counter['<unk>']
    
    return (unk_track_length/total_track_length), (unk_artist_length/total_artist_length)


def artist_frequency_statistics(train_data, val_data, test_data):
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
    second_common = artist_frequency_counter.most_common()[length:int(2*length)]
    third_common = artist_frequency_counter.most_common()[int(2*length):-1 * length]
    least_common = artist_frequency_counter.most_common()[-1 * length:]
    most_common_frequency = [item[1] for item in most_common]
    second_common_frequency = [item[1] for item in second_common]
    third_common_frequency = [item[1] for item in third_common]
    least_common_frequency = [item[1] for item in least_common]
    average_most_common_frequency = sum(most_common_frequency) / len(most_common_frequency)
    average_second_common_frequency = sum(second_common_frequency) / len(second_common_frequency)
    average_third_common_frequency = sum(third_common_frequency) / len(third_common_frequency)
    average_least_common_frequency = sum(least_common_frequency) / len(least_common_frequency)
    
    if args.dataset == 'melon':
        line_height = 1000
    elif args.dataset == 'million':
        line_height = 90

    popularity = [artist_frequency_counter[item] for item in artist_frequency_counter]
    outlier_length = int(len(artist_frequency_counter) * 0.01)
    popularity = [item[1] for item in artist_frequency_counter.most_common()[outlier_length:]]
    plt.figure(figsize=(8, 8))
    plt.hist(popularity, bins = 100)
    plt.vlines(x = most_common[-1][1],linestyle='dashed', ymin = 0, ymax = line_height, colors = 'red')
    plt.vlines(x = second_common[-1][1], linestyle='dashed', ymin = 0, ymax = line_height, colors = 'red')
    plt.vlines(x = third_common[-1][1], linestyle='dashed',ymin = 0, ymax = line_height, colors = 'red')


    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)

    figure_name = "./figure/" + args.dataset + "_" + "average_artist_frequency.png"
    plt.savefig(figure_name)
    plt.clf()
    
    return average_most_common_frequency, average_second_common_frequency, average_third_common_frequency, average_least_common_frequency

def track_frequency_statistics(train_data, val_data, test_data):
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
    second_common = track_frequency_counter.most_common()[length:int(2*length)]
    third_common = track_frequency_counter.most_common()[int(2*length):-1 * length]
    least_common = track_frequency_counter.most_common()[-1 * length:]
    most_common_frequency = [item[1] for item in most_common]
    second_common_frequency = [item[1] for item in second_common]
    third_common_frequency = [item[1] for item in third_common]
    least_common_frequency = [item[1] for item in least_common]
    average_most_common_frequency = sum(most_common_frequency) / len(most_common_frequency)
    average_second_common_frequency = sum(second_common_frequency) / len(second_common_frequency)
    average_third_common_frequency = sum(third_common_frequency) / len(third_common_frequency)
    average_least_common_frequency = sum(least_common_frequency) / len(least_common_frequency)
    
    popularity = [track_frequency_counter[item] for item in track_frequency_counter]
    outlier_length = int(len(track_frequency_counter) * 0.01)
    popularity = [item[1] for item in track_frequency_counter.most_common()[outlier_length:]]
    plt.figure(figsize=(8, 8))
    plt.hist(popularity, bins = 100)
    
    if args.dataset == 'melon':
        line_height = 1000
    elif args.dataset == 'million':
        line_height = 90

    plt.vlines(x = most_common[-1][1],linestyle='dashed', ymin = 0, ymax = line_height, colors = 'red')
    plt.vlines(x = second_common[-1][1], linestyle='dashed', ymin = 0, ymax = line_height, colors = 'red')
    plt.vlines(x = third_common[-1][1], linestyle='dashed',ymin = 0, ymax = line_height, colors = 'red')


    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)

    figure_name = "./figure/" + args.dataset + "_" + "average_track_frequency.png"
    plt.savefig(figure_name)
    plt.clf()
    
    return average_most_common_frequency, average_second_common_frequency, average_third_common_frequency, average_least_common_frequency



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="million", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    args = parser.parse_args()
    all = torch.load(os.path.join(args.dataset_dir, args.dataset, 'sets/filtered.pt'))
    train_data = torch.load(os.path.join(args.dataset_dir, args.dataset, 'sets/train.pt'))
    val_data = torch.load(os.path.join(args.dataset_dir, args.dataset, 'sets/val.pt'))
    test_data = torch.load(os.path.join(args.dataset_dir, args.dataset, 'sets/test.pt'))
    track_unk_proportion, artist_unk_proportion = unk_proportion(val_data, test_data)

    title_list = []
    title_token_list = []
    total_title_len = 0
    total_track_len = 0
    total_avg_chr_len = 0
    count = 0

    for item in tqdm(all):
        title = item['nrm_plylst_title']
        title_list.extend([title])
        
        tokens = title.split(" ")
        token_length = [len(x) for x in tokens]
        total_title_len += len(tokens)
        total_track_len += len(item['songs'])
        avg_chr_len = sum(token_length) / len(tokens)
        total_avg_chr_len += avg_chr_len
        
        title_token_list.extend(tokens)
        
    title_counter = Counter(title_list)
    token_counter = Counter(title_token_list)
    fa_1, fa_2, fa_3, fa_4 = artist_frequency_statistics(train_data, val_data, test_data)
    ft_1, ft_2, ft_3, ft_4 = track_frequency_statistics(train_data, val_data, test_data)
    print("the number of playlist after preprocessing is ", len(all))
    print("the ratio of train to validation, test data is ", len(train_data) * 100 /len(all) , " : " , (len(val_data) + len(test_data)) * 100/len(all))
    print("the proportion of unk track id tokens in validation and test sets is ", track_unk_proportion)
    print("the proportion of unk artist id tokens in validation and test sets is ", artist_unk_proportion)
    print("the number of unique title is ", len(title_counter))
    print("the number of unique word is ", len(token_counter))
    print("the average character length is ", total_avg_chr_len / len(all))
    print("the average title length is ", total_title_len / len(all))
    print("the average track length is ", total_track_len / len(all))
    print("the average f_t of the most, the second, the third, the least highest f_t group is ", ft_1, ft_2, ft_3, ft_4)
    print("the average f_a of the most, the second, the third, the least highest f_a group is ", fa_1, fa_2, fa_3, fa_4)

