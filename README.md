# Music Playlist Title Generation Using Artist Information

This repository provides the code for music playlist title generation model. For details, please refer to **Music Playlist Title Generation Using Artist Information**.

### Download Dataset

If you use the Melon Playlist Dataset, download the original data in ''./dataset/melon/data'' as shown below. 

* [dataset](./dataset)
    * [melon](./dataset/melon)
        * [data](./dataset/melon/data)
            * [song_meta.json](./dataset/melon/data/song_meta.json)
            * [test.json](./dataset/melon/data/test.json)
            * [train.json](./dataset/melon/data/train.json)
            * [val.json](./dataset/melon/data/val.json)

If you use the Million Playlist Dataset, download the original data in ''./dataset/million/data'' as shown below. 

* [dataset](./dataset)
    * [million](./dataset/million)
        * [data](./dataset/million/data)
            * [mpd.slice.0-999.json](./dataset/million/data/mpd.slice.0-999.json)
            .
            .
            .
            * [mpd.slice.999000-999999.json](./dataset/million/data/mpd.slice.999000-999999.json)


### Noise Filtering and Chronological Split
To filter out noisy data as suggested in the section 3 of the paper, run the following code.

'''sh
$ python preprocessing.py
'''

We provide the following parameters.

- `--dataset`: to choose between the Melon Playlist Dataset and the Million Playlist Dataset. E.g. "melon", "million"

- `--dataset_dir`: to set the directory where the data is stored. Default:"./dataset"; This means that the train, valid and test sets are stored at ''./dataset/{dataset_name}/sets'' and the tokenizers are stored at ''./dataset/{dataset_name}/tokenizer''. 

Note : 


### Acknowledgements

This repository includes code from the following repositories with modifications:
* [Music_Playlist_Title_Generation:_A_Machine_Learning_Approach](https://github.com/SeungHeonDoh/ply_title_gen)

* [Korean_BERT_Score](https://github.com/lovit/KoBERTScore)
