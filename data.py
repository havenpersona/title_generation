import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pickle

class MyDataset(Dataset):
    def __init__(self, split, dataset_dir, dataset, input_type, title_vocab, input_vocab, title_length, input_length):
        super().__init__()
        self.split = split
        self.dataset_dir = dataset_dir
        self.dataset = dataset
        self.input_type = input_type #'artist' or 'track'
        self.title_vocab = title_vocab.token_to_idx
        self.input_vocab = input_vocab.token_to_idx
        self.title_length = title_length
        self.input_length = input_length
        if self.split == "TRAIN":
            self.this_data = torch.load(os.path.join(self.dataset_dir, self.dataset, "sets/train.pt"))
        elif self.split == "VAL":
            self.this_data = torch.load(os.path.join(self.dataset_dir, self.dataset, "sets/val.pt"))
        elif self.split == "TEST":
            self.this_data = torch.load(os.path.join(self.dataset_dir, self.dataset, "sets/test.pt"))
        else:
            print("ERROR")
    def tokenize_title(self, title, title_vocab, title_length):
        
        tokens = ["<sos>"] + title.split() + ["<eos>"]
        tokens = [title_vocab[x] for x in tokens]
        output = torch.zeros(title_length, dtype=torch.long)
        if len(tokens) < title_length:
            output[:len(tokens)] = torch.tensor(tokens)
        else:
            output[:title_length - 1] = torch.tensor(tokens[:title_length - 1]) 
            output[-1] = tokens[-1]
        return output
    
    def tokenize_input(self, input, input_vocab, input_length):
        tokens = ["<sos>"] + input + ["<eos>"]
        tokens = [input_vocab[x] for x in tokens]
        output = torch.zeros(input_length, dtype=torch.long)
        if len(tokens) < input_length:
            output[:len(tokens)] = torch.tensor(tokens)
        else:
            output[:input_length - 1] = torch.tensor(tokens[:input_length - 1]) 
            output[-1] = tokens[-1]
        return output

    def __getitem__(self, index):
        item = self.this_data[index]
        pid = item['pid']
        title = item['nrm_plylst_title']
        title_seq = self.tokenize_title(title = title, title_vocab = self.title_vocab, title_length = self.title_length)
        if self.input_type == 'artist':
            artists = item['artist_ids']
            input_seq = self.tokenize_input(artists, self.input_vocab, self.input_length)
        elif self.input_type == 'track':
            tracks = item['songs']
            input_seq = self.tokenize_input(tracks, self.input_vocab, self.input_length)
        return input_seq, title_seq
        
    def __len__(self):
        output = self.this_data
        return len(output)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.dataset_dir = args.dataset_dir
        self.dataset = args.dataset
        self.input_type = args.input_type
        self.title_length = args.title_length
        self.input_length = args.input_length
        self.batch_size = args.batch_size

    def train_dataloader(self):
        title_vocab = pickle.load(open(os.path.join(self.dataset_dir, self.dataset, "tokenizer/title_split", self.dataset + "_vocab.pkl"), mode="rb"))
        input_vocab = pickle.load(open(os.path.join(self.dataset_dir, self.dataset, "tokenizer", self.input_type, self.dataset + "_vocab.pkl"), mode="rb"))

        train_set = MyDataset(split = 'TRAIN',
                    dataset_dir = self.dataset_dir,
                    dataset = self.dataset,
                    input_type = self.input_type,
                    title_vocab = title_vocab,
                    input_vocab = input_vocab,
                    title_length = self.title_length,
                    input_length = self.input_length
                    ) 
        train_data = torch.utils.data.DataLoader(train_set, batch_size = self.batch_size, shuffle=True, num_workers = 8)
        return train_data


    def val_dataloader(self):
        title_vocab = pickle.load(open(os.path.join(self.dataset_dir, self.dataset, "tokenizer/title_split", self.dataset + "_vocab.pkl"), mode="rb"))
        input_vocab = pickle.load(open(os.path.join(self.dataset_dir, self.dataset, "tokenizer", self.input_type, self.dataset + "_vocab.pkl"), mode="rb"))
        
        val_set = MyDataset(split = 'VAL',
                    dataset_dir = self.dataset_dir,
                    dataset = self.dataset,
                    input_type = self.input_type,
                    title_vocab = title_vocab,
                    input_vocab = input_vocab,
                    title_length = self.title_length,
                    input_length = self.input_length
                    ) 
        val_data = torch.utils.data.DataLoader(val_set, batch_size = self.batch_size, shuffle=False, num_workers = 8)
        return val_data

    def test_dataloader(self):
        title_vocab = pickle.load(open(os.path.join(self.dataset_dir, self.dataset, "tokenizer/title_split", self.dataset + "_vocab.pkl"), mode="rb"))
        input_vocab = pickle.load(open(os.path.join(self.dataset_dir, self.dataset, "tokenizer", self.input_type, self.dataset + "_vocab.pkl"), mode="rb"))
    
        test_set = MyDataset(split = 'TEST',
                    dataset_dir = self.dataset_dir,
                    dataset = self.dataset,
                    input_type = self.input_type,
                    title_vocab = title_vocab,
                    input_vocab = input_vocab,
                    title_length = self.title_length,
                    input_length = self.input_length
                    ) 
        test_data = torch.utils.data.DataLoader(test_set, batch_size = self.batch_size, shuffle=False, num_workers = 8)
        return test_data


