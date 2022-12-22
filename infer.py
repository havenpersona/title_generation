import pickle
import os
import torch
import json
import numpy as np
from rouge import Rouge
from model import Transformer
from argparse import ArgumentParser, Namespace
from torchmetrics import BLEUScore
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize 
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from KoreanBERTScore.score import BERTScore
from evaluate import load #pip install bert-score
from tqdm import tqdm
from collections import Counter


test_file_name = "test"
ckpt_name = "million_track_2022-12-22 11:17:20.830998"
save_path = "./inference/" + ckpt_name + "_" +  test_file_name


      

def tokenize_title(title, title_vocab, title_length):
    tokens = ["<sos>"] + title.split() + ["<eos>"]
    tokens = [title_vocab.token_to_idx[x] for x in tokens]
    output = torch.zeros(title_length, dtype=torch.long)
    if len(tokens) < title_length:
        output[:len(tokens)] = torch.tensor(tokens)
    else:
        output[:title_length - 1] = torch.tensor(tokens[:title_length - 1]) 
        output[-1] = tokens[-1]
    return output

def tokenize_input(input, input_vocab, input_length):
    tokens = ["<sos>"] + input + ["<eos>"]
    tokens = [input_vocab.token_to_idx[x] for x in tokens]
    output = torch.zeros(input_length, dtype=torch.long)
    if len(tokens) < input_length:
        output[:len(tokens)] = torch.tensor(tokens)
    else:
        output[:input_length - 1] = torch.tensor(tokens[:input_length - 1]) 
        output[-1] = tokens[-1]
    return output

def generate_title(input_type, model, title_vocab, input_vocab, tracks, artists, title_length, input_length, device):
    sos_token = tokenize_title('', title_vocab, title_length)
    sos_token = torch.unsqueeze(sos_token, 0)[:,0]
    target = [sos_token]
    if input_type == 'track':
        track_sequence = tokenize_input(tracks, input_vocab, input_length).to(device)
        input_sequence = torch.unsqueeze(track_sequence, 0)
    elif input_type == 'artist':
        artist_sequence = tokenize_input(artists, input_vocab, input_length).to(device)
        input_sequence = torch.unsqueeze(artist_sequence, 0)
    
    
    
    source_mask = model.get_source_mask(input_sequence)
    encoder_source = model.encoder(input_sequence, source_mask)
    
    max_length = 64
    for x in range(1, max_length):
        with torch.no_grad():
            target_tensor = torch.LongTensor(target).unsqueeze(0).to(device) 
            target_mask = model.get_target_mask(target_tensor)
            output, attention = model.decoder(target_tensor, encoder_source, target_mask, source_mask)
        current_token = output.argmax(2)
        current_token = current_token[:, -1].item()
        target.append(current_token)
        if current_token == 2:
            break
    result = [title_vocab.idx_to_token[x] for x in target[1:-1]]
    return result

    #target_mask = get_target_mask(target)
    
    #output, attention = self.decoder(target, encoder_source, target_mask, source_mask)
  
def main(args):
    output_vocab = pickle.load(open(os.path.join(args.dataset_dir, args.dataset, "tokenizer/title_split", args.dataset + "_vocab.pkl"), mode="rb"))
    input_vocab = pickle.load(open(os.path.join(args.dataset_dir, args.dataset, "tokenizer", args.input_type, args.dataset + "_vocab.pkl"), mode="rb"))
    bleu_uni = BLEUScore(n_gram=1)
    bleu_bi = BLEUScore(n_gram=2)
    rouge = Rouge()

    if args.dataset == 'melon':
        model_name = "beomi/kcbert-base"
        bertscore = BERTScore(model_name, best_layer=4)
        word_embedding_model = models.Transformer(
            model_name_or_path="klue/roberta-small", 
            max_seq_length=64,
            do_lower_case=True
        )   
        pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    elif args.dataset == 'million':
        bertscore = load("bertscore")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


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
    
    checkpoint = torch.load(os.path.join("./checkpoint", ckpt_name + ".ckpt"), map_location=torch.device(f"cuda:{args.gpus}"))
    state_dict = checkpoint.get("state_dict")
    #new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
    #new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
    model.load_state_dict(state_dict)
    
    model = model.to(f"cuda:{args.gpus}")
    model.eval()
    test_path = (os.path.join(args.dataset_dir, args.dataset, "sets/" + test_file_name + ".pt"))
    test_file = torch.load(test_path)


    inference = {}
    bleu_uni_list = []
    bleu_bi_list = []
    rouge_uni_list = []
    rouge_bi_list = []
    meteor_list = []
    bert_list = []
    sbert_list = []
    unigram_list = []
    bigram_list = []
    trigram_list = []

    for item in tqdm(test_file):
        result = generate_title(args.input_type, model, output_vocab, input_vocab, item['songs'], item['artist_ids'], args.title_length, args.input_length, args.gpus)
        unigram_list.extend(result)
        line = " ".join(result)
        token = word_tokenize(line)
        bigram = list(ngrams(token, 2)) 
        trigram = list(ngrams(token, 3))
        bigram_list.extend(bigram)
        trigram_list.extend(trigram)
        
        preds = [" ".join(result)]
        target = [[item['original_title']]]
        bleu_uni_score = bleu_uni(preds,target)
        bleu_bi_score = bleu_bi(preds,target)

        hypothesis = " ".join(result)
        reference = item['original_title']
        rouge_score = rouge.get_scores(hypothesis, reference)
        rouge_uni_score = rouge_score[0]["rouge-1"]["f"]
        rouge_bi_score = rouge_score[0]["rouge-2"]["f"]

        test = [result]
        pred = item['original_title'].split(" ")
        meteor = meteor_score(test, pred)
        
        candidates = [" ".join(result)]
        references = [item['original_title']]
        if args.dataset == 'melon':
            bert = bertscore(references, candidates)
        elif args.dataset == 'million':
            bert = bertscore.compute(predictions=candidates, references=references, lang="en")

        embeddings1 = sentence_model.encode(candidates, convert_to_tensor=True)
        embeddings2 = sentence_model.encode(references, convert_to_tensor=True)
        sbert = util.cos_sim(embeddings1, embeddings2)

        
        bleu_uni_list.append(bleu_uni_score.item())
        bleu_bi_list.append(bleu_bi_score.item())
        rouge_uni_list.append(rouge_uni_score)
        rouge_bi_list.append(rouge_bi_score)
        meteor_list.append(meteor)
        if args.dataset == 'melon':
            this_bert_score = bert[0]
            bert_list.append(this_bert_score)
        elif args.dataset == 'million':
            this_bert_score = bert['f1'][0]
            bert_list.append(this_bert_score)
        sbert_list.append(sbert[0].item())
        inference[item['pid']] = {
            "ground_truth": item['original_title'],
            "prediction": " ".join(result),
            "bleu uni" : bleu_uni_score.item(),
            "bleu bi" : bleu_bi_score.item(),
            "rouge uni" : rouge_uni_score,
            "rouge bi" : rouge_bi_score,
            "meteor" : meteor,
            "bert" : this_bert_score,
            "sbert" : sbert[0].item()
        }
        
    inference['summary'] = {
        "BLEU Unigram(Average)" : np.mean(bleu_uni_list),
        "BLEU Bigram(Average)" : np.mean(bleu_bi_list),
        "Rouge Unigram(Average)" : np.mean(rouge_uni_list),
        "Rouge Bigram(Average)" : np.mean(rouge_bi_list), 
        "Meteor(Average)" : np.mean(meteor_list),
        "BERT (Average)"  : np.mean(bert_list),
        "SBERT (Average)"  : np.mean(sbert_list),
        "distinct - 1" : len(Counter(unigram_list))/len(unigram_list),
        "distinct - 2" : len(Counter(bigram_list))/len(bigram_list),
        "distinct - 3" : len(Counter(trigram_list))/len(trigram_list)
        }
    print("***" + test_file_name + " ***")
    print("BLEU Unigram(Average)", np.mean(bleu_uni_list))
    print("BLEU Bigram(Average)", np.mean(bleu_bi_list))
    print("Rouge Unigram(Average)", np.mean(rouge_uni_list))
    print("Rouge Bigram(Average)", np.mean(rouge_bi_list))
    print("Meteor(Average)", np.mean(meteor_list))
    print("BERT (Average)", np.mean(bert_list))
    print("SBERT (Average)", np.mean(sbert_list))
    print("distinct - 1", len(Counter(unigram_list))/len(unigram_list))
    print("distinct - 2", len(Counter(bigram_list))/len(bigram_list))
    print("distinct - 3", len(Counter(trigram_list))/len(trigram_list))
    if not os.path.exists(os.path.join("./inference", ckpt_name)):
        os.makedirs(os.path.join("./inference", ckpt_name))
    with open(os.path.join("./inference", ckpt_name, test_file_name + ".json"), mode="w", encoding='utf-8') as io:
        json.dump(inference, io, ensure_ascii=False, indent=4)
  
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="melon", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--input_type", default="artist", type=str)
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--model_size", default=128, type=int)
    parser.add_argument("--inner_size", default=256, type=int)
    parser.add_argument("--encoder_layers", default=3, type=int)
    parser.add_argument("--decoder_layers", default=3, type=int)
    parser.add_argument("--n_heads", default=8, type=int)
    parser.add_argument("--dropout", default=0.1, type=int)
    parser.add_argument("--max_length", default=64, type=int)
    parser.add_argument("--input_length", default=32, type=int)
    parser.add_argument("--title_length", default=32, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    
    args = parser.parse_args()
    
    main(args)