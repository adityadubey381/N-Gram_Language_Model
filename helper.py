import torch
import random
import re
from collections import defaultdict

from nltk.tokenize import word_tokenize



# Prepare Dataset

def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# Text Processing

def tokenize_text(text):
    # for new lines
    text = text.replace("\n", " ")

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # tokenize
    tokens = text.split()

    return tokens

# buling the Vocb
def build_vocab(tokens):
    vocab = sorted(set(tokens))
    word_to_idx = {word:i for i,word in enumerate(vocab)}
    idx_to_word = {i:word for word,i in word_to_idx.items()}

    return vocab , word_to_idx , idx_to_word

# Build an NGram Model
def train_model(tokens, word_to_idx):

    vocab_size = len(word_to_idx)

    matrix = torch.zeros((vocab_size, vocab_size))
    for i in range(len(tokens)-1):
        w1 = tokens[i]
        w2 = tokens[i+1]

        idx1 = word_to_idx[w1]
        idx2 = word_to_idx[w2]

        matrix[idx1][idx2] += 1
    prob_matrix = matrix / matrix.sum(dim=1, keepdim=True)

    return prob_matrix


# fext generation function
def generate_text(prob_matrix, idx_to_word, word_to_idx, length=10):

    vocab_size = len(idx_to_word)

    current_idx = random.randint(0, vocab_size-1)
    sentence = [idx_to_word[current_idx]]
    for _ in range(length):

        probs = prob_matrix[current_idx]

        next_idx = torch.multinomial(probs,1).item()
        sentence.append(idx_to_word[next_idx])

        current_idx = next_idx

    return " ".join(sentence)
