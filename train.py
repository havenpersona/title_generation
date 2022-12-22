import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace
import pickle
import os
import torch
from data import MyDataModule
from model import Transformer
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime

save_path = './checkpoint'


def main(args):
    input_vocab = pickle.load(open(os.path.join(args.dataset_dir, args.dataset, "tokenizer", args.input_type, args.dataset + "_vocab.pkl"), mode="rb"))
    output_vocab = pickle.load(open(os.path.join(args.dataset_dir, args.dataset, "tokenizer/title_split", args.dataset + "_vocab.pkl"), mode="rb"))
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename=args.dataset + '_' + args.input_type + '_' + str(datetime.now()),
        save_top_k=1,
        save_last= False,
        monitor="val_loss",
        mode='min',
        save_weights_only=True,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=20, verbose=True, mode="min"
    )

    data_module = MyDataModule(args)
    model = Transformer(input_size = len(input_vocab), 
                    output_size = len(output_vocab), 
                    model_size = args.model_size, 
                    inner_size = args.inner_size,
                    encoder_layers = args.encoder_layers, 
                    decoder_layers = args.decoder_layers, 
                    n_heads = args.n_heads, 
                    dropout = args.dropout, 
                    max_length = args.max_length, 
                    device = args.gpus,
                    lr = args.lr, 
                    weight_decay = args.weight_decay)
    trainer = pl.Trainer(callbacks=[
                            early_stop_callback,
                            checkpoint_callback
                        ],
                        max_epochs= args.max_epochs,
                        gpus = [args.gpus])
    trainer.fit(model, data_module)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="melon", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--input_type", default="artist", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--model_size", default=128, type=int)
    parser.add_argument("--inner_size", default=256, type=int)
    parser.add_argument("--encoder_layers", default=3, type=int)
    parser.add_argument("--decoder_layers", default=3, type=int)
    parser.add_argument("--n_heads", default=8, type=int)
    parser.add_argument("--dropout", default=0.1, type=int)
    parser.add_argument("--max_length", default=64, type=int)
    parser.add_argument("--title_length", default=32, type=int)
    parser.add_argument("--input_length", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=float)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--max_epochs", default=100, type=float)
    parser.add_argument("--gpus", default=0, type=int)
    args = parser.parse_args()

    main(args)

 