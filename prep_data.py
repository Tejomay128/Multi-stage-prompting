from datasets import load_dataset


def sort_input_by_length(input_data):
    lengths = [(idx, len(line.strip().split())) for idx, line in enumerate(input_data)]
    sorted_lengths = sorted(lengths, key=lambda x: x[1], reverse=True)
    sorted_input_data = []
    for idx,_ in sorted_lengths:
        sorted_input_data.append(input_data[idx])
    return sorted_input_data


def pad(data):
    


def get_eng_hi_dataset():
    dataset = load_dataset("cfilt/iitb-english-hindi")
    input_data = dataset['en']
    sorted_input_data = sort_input_by_length(input_data)



