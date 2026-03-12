from helper import load_dataset
from helper import tokenize_text
from helper import build_vocab
from helper import train_model
from helper import generate_text

import streamlit as st

st.title("N-Gram Model(Bigram)")
st.write("Simple AI text generator using N-Gram model")

text = load_dataset("datasets.txt")
tokens = tokenize_text(text)
vocab, word_to_idx, idx_to_word  = build_vocab(tokens)
prob_matrix = train_model(tokens, word_to_idx)
length = st.slider("Sentence Length", 5, 20, 10)
num_sentences = st.slider("Number of Sentences", 1, 20, 5)

if st.button("Generate Sentence"):
    for i in range(num_sentences):
        sentence = generate_text(prob_matrix, idx_to_word, word_to_idx, length)

        st.write(f"{i+1}. {sentence}")
