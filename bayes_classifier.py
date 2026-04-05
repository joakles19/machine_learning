import numpy as np
import pandas as pd
from collections import Counter


class TextClassifier:
    def __init__(self, dataset: pd.DataFrame, alpha=1):
        self.data = dataset.dropna()
        self.alpha = alpha
        self.category_col = self.data.columns[0]
        self.text_col = self.data.columns[1]
        
        self.priors = self.__priors(self.data)
        self.likelihoods, self.vocab_size = self.__likelihoods(self.data)
        
    def __group_words(self, data: pd.DataFrame):
        grouped = (
            data.groupby(self.category_col)[self.text_col]
            .agg(' '.join)
            .reset_index()
        )

        return grouped
    
    def __priors(self, data: pd.DataFrame):
        doc_counts = data[self.category_col].value_counts()
        total_docs = len(data)
        priors = doc_counts / total_docs

        return priors
    
    def __likelihoods(self, data: pd.DataFrame):
        grouped = self.__group_words(data)
        likelihoods = {}
        vocab = set()
        
        for _, row in grouped.iterrows():
            cat = row[self.category_col]
            words = row[self.text_col].split()
            word_counts = Counter(words)
            likelihoods[cat] = word_counts
            vocab.update(word_counts.keys())
        
        vocab_size = len(vocab)
        
        for cat in likelihoods:
            total_words = sum(likelihoods[cat].values())
            likelihoods[cat] = {
                word: (count + self.alpha) / (total_words + self.alpha * vocab_size)
                for word, count in likelihoods[cat].items()
            }
            for word in vocab:
                if word not in likelihoods[cat]:
                    likelihoods[cat][word] = self.alpha / (total_words + self.alpha * vocab_size)
        
        return likelihoods, vocab_size
    
    def predict(self, text: str):
        words = text.split()
        posteriors = {}
        
        for cat in self.priors.index:
            log_prob = np.log(self.priors[cat])
            
            for word in words:
                word_prob = self.likelihoods[cat].get(word, self.alpha / 
                                (sum(self.likelihoods[cat].values()) + self.alpha * self.vocab_size))
                log_prob += np.log(word_prob)
            
            posteriors[cat] = log_prob
        
        max_log = max(posteriors.values())
        probs = {cat: np.exp(logp - max_log) for cat, logp in posteriors.items()}

        total = sum(probs.values())
        posteriors_normalized = {cat: p/total for cat, p in probs.items()}
        
        predicted = max(posteriors_normalized, key=posteriors_normalized.get)
        return predicted