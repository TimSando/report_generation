import json
import pickle
import random
import re
from typing import List, Tuple
from nptyping import NDArray

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import load


class GenerateData:
    def __init__(self):
        pass

    def generate_training_data(self, filename: str):
        loaded_data = self.load_data(filename)
        sequenced_test_data = self.create_sequence(loaded_data)
        self.separate_test_data(sequenced_test_data)
        self.word_encode(loaded_data)
        return self.input_int, self.target_int

    def load_data(self, filename: str):
        with open(f"data/{filename}", "rb") as f:
            loaded_data = pickle.load(f)
            # cleaning the text by removing most non-letter or number characters to ensure the final model is focusing on the right things
            loaded_data = [re.sub("[^A-z' 0-9]", "", i) for i in loaded_data]
        return loaded_data

    def create_sequence(self, input_text: list, sequence_length: int = 5) -> List[str]:
        """Reduces the list of longer strings of text down to strings that are {sequence_length} long to create more manageable training data.

        Args:
            input_text (list): A list of strings that will be the curated training data. Each string will correlate to a report
            sequence_length (int, optional): Length of the output sequence string. Defaults to 5.

        Returns:
            list[str]: A list of strings with length {sequence_length}
        """
        test_data = []
        sequences = []
        for entry in input_text:
            if len(entry.split()) > sequence_length:
                for i in range(sequence_length, len(entry.split())):
                    text_sequence = entry.split()[i - sequence_length : i + 1]
                    sequences.append(" ".join(text_sequence))
            else:
                sequences = [entry]
            # # TODO: Remove this break to get everything
            # break
        test_data.append(sequences)
        test_data = sum(test_data, [])
        return test_data

    def separate_test_data(self, test_data: list):
        """Splitting out the test data into input and target data that will be used for training the model.

        Args:
            test_data (list): A list of strings that are defined by the sequence length variable

        Returns:
            input_data: List of strings that remove the last word in the sequence
            target_data: List of strings that include the final word but remove the first word.
        """
        self.input_data = []
        self.target_data = []
        for sequence in test_data:
            self.input_data.append(" ".join(sequence.split()[:-1]))
            self.target_data.append(" ".join(sequence.split()[1:]))

    def word_encode(self, full_training_data: List[str]):
        self.word_token = {}
        count = 0
        for word in set(" ".join(full_training_data).split()):
            self.word_token[count] = word
            count += 1
        self.token_word = {t: w for w, t in self.word_token.items()}
        # Convert the sequenced strings into strings of integers as defined by the token -> word mapping
        self.input_int = np.array([self.get_integer_sequence(s) for s in self.input_data])
        self.target_int = np.array([self.get_integer_sequence(s) for s in self.target_data])

    def get_integer_sequence(self, sequence: str):
        return [self.token_word[word] for word in sequence.split()]


if __name__ == "__main__":
    data = GenerateData()
    input_data, training_data = data.generate_training_data("plots_text.pickle")
    print(input_data.shape[0])
