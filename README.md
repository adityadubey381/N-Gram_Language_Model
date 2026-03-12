# N-Gram Language Model

## Project Overview

This project implements a simple **N-Gram language model** in Python.
The goal of the project is to understand how basic language models work and how a system can predict the **next word in a sentence** using probability.

The program reads a dataset, processes the text, builds n-grams, and predicts the next word based on the previous words.

---

## How the Program Works

### 1. Text Preprocessing

The dataset text is cleaned before building the model.
This includes:

* Removing numbers
* Removing punctuation
* Replacing new lines with spaces

After cleaning, the text is split into words (tokens).

### 2. Creating N-Grams

The tokens are used to create sequences of words called **n-grams**.

Example sentence:
I love machine learning

Bigram example:

* (I, love)
* (love, machine)
* (machine, learning)

These sequences help the model learn which words usually come next.

### 3. Probability Calculation

The program counts how many times a word appears after another word.
Using these counts, it calculates the probability of the next word.

### 4. Word Prediction

When a user enters a sentence, the model predicts the next word based on the previous words.
The prediction is selected using probability sampling.

---

## Files in This Project

* `app.py` – main program to run the model
* `helper.py` – helper functions used in the project
* `dataset.txt` – dataset used to train the model
* `ngrams.ipynb` – notebook used for experimentation
* `requirements.txt` – project dependencies

---

## How to Run the Project

1. Clone the repository
2. Install the required packages
3. Run the main file

Example:

python app.py

---

## Technologies Used

* Python
* PyTorch
* Regular Expressions
* N-Gram Language Model

---

## Conclusion

This project helped in understanding how traditional language models work.
Although modern NLP models use deep learning, N-Gram models are a good starting point for learning text prediction and language modeling.
