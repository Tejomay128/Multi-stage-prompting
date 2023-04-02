from datasets import load_dataset
from transformers import GPT2Tokenizer


# sort the entire corpus based on input length (to minimize padding)
def sort_input_by_length(dataset, lang1='en'):
    symbols = ['<', '>']
    lengths = [(idx, len(line[lang1].strip().split())) for idx, line in enumerate(dataset)]
    sorted_lengths = sorted(lengths, key=lambda x: x[1], reverse=True)
    sorted_dataset = []
    for i, (idx,_) in enumerate(sorted_lengths):
        # print(f"En: {dataset[idx][lang1]} || Hi: {dataset[idx]['hi']}")
        flag = False
        for symbol in symbols:
            if symbol in dataset[idx][lang1]:
                flag = True
                break
        if len(dataset[idx][lang1]) > 0 and not flag:
            sorted_dataset.append(dataset[idx])
    return sorted_dataset

#filters out sentences that won't fit within the model's token limit
def filter_dataset(tokenizer, token_limit, len_prefix, dataset, lang1='en', lang2='hi'):
    new_dataset = []
    for pair in dataset:
        if len_prefix + len(pair[lang1]) + len(pair[lang2]) < token_limit:
            tok1 = tokenizer(pair[lang1])['input_ids']
            tok2 = tokenizer(pair[lang2])['input_ids']
            if len_prefix + len(tok1) + len(tok2) < token_limit:
                new_dataset.append(pair)
    return new_dataset


def get_eng_hi_dataset():
    dataset = load_dataset("cfilt/iitb-english-hindi")
    # train_data = filter_dataset(tokenizer, token_limit, len_prefix, dataset['train']['translation'])
    # test_data =filter_dataset(tokenizer, token_limit, len_prefix, dataset['test']['translation'])
    val_data = dataset['validation']['translation']
    test_data = dataset['test']['translation']

    val_data = sort_input_by_length(val_data)
    test_data = sort_input_by_length(test_data)

    return val_data, test_data
