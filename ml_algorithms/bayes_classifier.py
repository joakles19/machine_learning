import numpy as np
import pandas as pd
from collections import Counter

class TextClassifier:
    def __init__(self, dataset: pd.DataFrame, alpha=1, ngram=3):
        self.data = dataset.dropna()
        self.alpha = alpha
        self.ngram = ngram
        self.category_col = self.data.columns[0]
        self.text_col = self.data.columns[1]

        self.priors = self.__compute_priors()
        self.likelihoods, self.vocab, self.cat_totals = self.__compute_likelihoods()

        # Precompute log values
        self.log_priors = {cat: np.log(prob) for cat, prob in self.priors.items()}
        self.log_likelihoods = {}
        self.log_unknown = {}

        vocab_size = len(self.vocab)
        for cat in self.likelihoods:
            total_words = self.cat_totals[cat] + self.alpha * vocab_size
            self.log_likelihoods[cat] = {word: np.log(prob) for word, prob in self.likelihoods[cat].items()}
            self.log_unknown[cat] = np.log(self.alpha / total_words)

    def __group_words(self, data: pd.DataFrame):
        return data.groupby(self.category_col)[self.text_col].agg(' '.join).reset_index()
    
    def __get_n_grams(self, words, n):
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    def __compute_priors(self):
        doc_counts = self.data[self.category_col].value_counts()
        total_docs = len(self.data)
        return (doc_counts / total_docs).to_dict()

    def __compute_likelihoods(self):
        grouped = self.__group_words(self.data)
        likelihoods = {}
        vocab = set()
        cat_totals = {}

        for _, row in grouped.iterrows():
            cat = row[self.category_col]
            words = row[self.text_col].split()
            
            # Include n-grams
            all_words = words.copy()
            for n in range(2, self.ngram+1):
                all_words += self.__get_n_grams(words, n)
            
            word_counts = Counter(all_words)
            likelihoods[cat] = word_counts
            vocab.update(word_counts.keys())
            cat_totals[cat] = sum(word_counts.values())

        vocab_size = len(vocab)
        for cat in likelihoods:
            total_words = cat_totals[cat]
            likelihoods[cat] = {word: (count + self.alpha) / (total_words + self.alpha * vocab_size)
                                for word, count in likelihoods[cat].items()}
            for word in vocab:
                if word not in likelihoods[cat]:
                    likelihoods[cat][word] = self.alpha / (total_words + self.alpha * vocab_size)

        return likelihoods, vocab, cat_totals

    def predict(self, texts: pd.Series):
        results = []

        for text in texts:
            words = text.split()
            
            all_words = words.copy()
            for n in range(2, self.ngram+1):
                all_words += self.__get_n_grams(words, n)

            posteriors = {}
            for cat in self.priors:
                log_prob = self.log_priors[cat]
                log_prob += sum(self.log_likelihoods[cat].get(word, self.log_unknown[cat]) for word in all_words)
                posteriors[cat] = log_prob

            predicted = max(posteriors, key=posteriors.get)
            results.append(predicted)

        return results