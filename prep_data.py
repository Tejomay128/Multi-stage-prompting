from datasets import load_dataset
import torch
from transformers import MT5TokenizerFast


def sort_input_by_length(input_data):
    lengths = [(idx, len(line.strip().split())) for idx, line in enumerate(input_data)]
    sorted_lengths = sorted(lengths, key=lambda x: x[1], reverse=True)
    sorted_input_data = []
    for idx,_ in sorted_lengths:
        sorted_input_data.append(input_data[idx])
    return sorted_input_data


def get_eng_hi_dataset():
    dataset = load_dataset("cfilt/iitb-english-hindi")
    input_data = dataset['en']
    sorted_input_data = sort_input_by_length(input_data)
    return sorted_input_data


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
