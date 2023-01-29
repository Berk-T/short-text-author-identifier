import os
import re

import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

from collections import Counter

def preprocess_data(chat_dir):
    chat_path = os.path.join(chat_dir, "_chat.txt")
    out_path = os.path.join(chat_dir, "_chat_processed.txt")
    senders_dict = dict()
    with open(chat_path, "r", encoding="utf8") as infile, open(out_path, "w", encoding="utf8") as outfile:
        prev_sender = ""
        group_name = None
        for line in infile:
            line = line.strip()
            line = line.lower()

            sender = re.findall(
                r'(?<=\[[0-9:\., ]{18}\] )[^:]*(?=:)', line)

            if group_name is None and sender:
                group_name = sender[0].replace(":", "")

            if remove_line(line, group_name):
                continue

            if sender:
                sender = sender[0].replace(":", "")
                prev_sender = sender

                if sender not in senders_dict.keys():
                    senders_dict[sender] = len(senders_dict)

                text = line[23 + len(sender):]
                if len(text) == 0:
                    continue

                new_line = sender + ':' + text + "\n"
            else:
                if len(line) == 0:
                    continue
                new_line = prev_sender + ':' + line + "\n"
            outfile.write(new_line)
    return senders_dict


def remove_line(line, group_name):
    omit_types = ["audio", "image", "GIF", "sticker",
                  "video", "contact card", "document"]
    omit_messages = [type.lower() + " omitted" for type in omit_types]
    system_messages = [group_name + ":", "created group", "changed the subject to", "changed the group description", "no longer an admin", "now an admin", "changed this group's icon",
                       "messages and calls are end-to-end encrypted. no one outside of this chat, not even whatsapp, can read or listen to them."]
    location_messages = ["location:", "see my real-time location on maps:"]

    remove_filters = system_messages + location_messages + omit_messages

    return any(filter in line for filter in remove_filters)


def tokenize(text):
    return [s.lower() for s in re.split(r'\W+', text) if len(s) > 0]

def create_vocab(chat_dir, sender_indices, threshold=1):
    data_path = os.path.join(chat_dir, "_chat_processed.txt")
    lines = []
    tokenized_data = []
    with open(data_path, "r", encoding="utf8") as chat:
        lines = chat.readlines()

    for line in lines:
        colon_index = line.find(':')
        sender = line[:colon_index]
        text = line[colon_index + 1:]
        tokenized_data.append((tokenize(text), sender_indices[sender]))

    freqs = Counter()
    for tokens, _ in tokenized_data:
        freqs.update(tokens)
    vocab = {'<eos>': 0, '<unk>': 1}
    for token, freq in freqs.items():
        if(freq <= threshold):
            continue
        vocab[token] = len(vocab)
    return vocab, tokenized_data, lines

class ChatDataset(Dataset):
    def __init__(self, data):
        self.data = []
        for d in data:
            if len(d[1]) == 0 or len(d[2]) == 0:
                continue
            self.data.append(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        _, _, indices, label = self.data[i]
        return {
            'data': torch.tensor(indices).long(),
            'label': torch.tensor(label).long()
        }