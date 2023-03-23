from datasets import load_dataset
import torch
from transformers import MT5Tokenizer


# sort the entire corpus based on input length (to minimize padding)
def sort_input_by_length(dataset):
    input_data = dataset['en']
    lengths = [(idx, len(line.strip().split())) for idx, line in enumerate(input_data)]
    sorted_lengths = sorted(lengths, key=lambda x: x[1], reverse=True)
    sorted_dataset = {'en':[], 'hi':[]}
    for idx,_ in lengths:
        sorted_dataset['en'].append(dataset['en'][idx])
        sorted_dataset['hi'].append(dataset['hi'][idx])
    return sorted_dataset


def get_eng_hi_dataset():
    dataset = load_dataset("cfilt/iitb-english-hindi")
    sorted_input_data = sort_input_by_length(dataset)
    return sorted_input_data


# pad input ids to make sentence length equal in a batch, make corresponding attention masks
def pad(input_ids):
    max_length = 0
    for x in input_ids:
        max_length = max(max_length, len(x))
    
    input_masks = []
    for i, x in enumerate(input_ids):
        mask = []
        for i in range(max_length):
            if i < len(x):
                mask.append(1)
            else:
                input_ids[i].append(0)
                mask.append(0)
    
    return input_ids, input_masks


def tokenize(sents):
    tokenizer = MT5Tokenizer.from_pretrained("THUMT/mGPT")
    input_ids = tokenizer(sents)['input_ids']
    input_ids, input_masks = pad(input_ids)
    return input_ids, input_masks