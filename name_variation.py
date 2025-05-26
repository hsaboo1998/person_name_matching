from typing import Tuple, Union
from pandas import DataFrame as pd_df
import numpy as np
import re
from constants import LENGTHWISE_ERROR_DISTRIBUTION, KEYBOARD_LAYOUT
from tqdm import tqdm
tqdm.pandas()

class NameVariationGenerator:
    def __init__(self):
        self.variations_dict = {
            'se': self.make_space_error,
            'he': self.make_hyphen_error,
            'ab': self.make_abbreviation,
            'wj': self.make_words_jumble,
            'wm': self.make_word_miss,
            'kte': self.make_keyboard_typing_error,
            'tse': self.make_truncated_string_error,
        }
        self.lengthwise_error_distribution = LENGTHWISE_ERROR_DISTRIBUTION
        self.keyboard_layout = np.array(KEYBOARD_LAYOUT)
        
    def generate_max_error_lengths(self, count: int):
        max_error_lengths = list(self.lengthwise_error_distribution.keys())
        prob_max_error_lengths = list(self.lengthwise_error_distribution.values())
        return np.random.choice(a=max_error_lengths, size=count, p=prob_max_error_lengths)
    
    @staticmethod
    def rectify_max_error_length(name: str, max_error_length: int):
        if 0.3*len(name) < max_error_length:
            max_error_length = int(0.3*len(name))
        return max_error_length
    
    def make_space_error(self, name: str, max_error_length=None):
        return name.replace(' ', '', 1)
    
    def make_hyphen_error(self, name: str, max_error_length=None):
        if '-' in name:
            name = name.replace('-', ' ', 1)
        if ' ' in name:
            name = name.replace(' ', '-', 1)
        return name
    
    def make_abbreviation(self, name: str, max_error_length=None):
        parts_of_name = name.split()
        if len(parts_of_name) > 1:
            pos_abbr = np.random.choice(range(len(parts_of_name)))
            parts_of_name[pos_abbr] = parts_of_name[pos_abbr][0]
        return ' '.join(parts_of_name)
    
    def make_words_jumble(self, name: str, max_error_length=None):
        parts_of_name = name.split()
        if len(set(parts_of_name)) > 1:
            while parts_of_name==name.split():
                np.random.shuffle(parts_of_name)
        return ' '.join(parts_of_name)
    
    def make_word_miss(self, name: str, max_error_length: int):
        parts_of_name = name.split()
        if len(parts_of_name) > 1:
            i = np.random.choice(len(parts_of_name))
            parts_of_name[i] = ''
        return ' '.join(parts_of_name)
    
    def make_truncated_string_error(self, name: str, max_error_length: int):
        error_length = self.rectify_max_error_length(name, max_error_length)
        if error_length>0:
            name = name[:-error_length]
        return name
    
    @property
    def generate_keyboard_probability_matrix(self):
        def dist(a,b):
            if (a==b).all():
                return np.inf
            return np.linalg.norm(a-b)**2
        keyboard_position_letters = [np.argwhere(self.keyboard_layout==chr(uc))[0] for uc in range(97, 123)]
        score_matrix = np.array([[1/dist(a,b) for b in keyboard_position_letters] for a in keyboard_position_letters])
        probability_matrix = score_matrix/np.sum(score_matrix, axis=1).reshape(-1,1)
        return probability_matrix
    
    def make_keyboard_typing_error(self, name: str, max_error_length: int):
        error_length = self.rectify_max_error_length(name, max_error_length)
        letters = list(name)
        if error_length>0:
            error_positions = np.random.choice(np.arange(len(letters)), size=error_length)
            probability_matrix = self.generate_keyboard_probability_matrix
            for i in np.arange(len(letters)):
                letter = letters[i]
                if i in error_positions and re.search('[^a-z]', letter) is None:
                    unicode_letter = np.random.choice(np.arange(97, 123), p=probability_matrix[ord(letter)-97])
                    letters[i] = chr(unicode_letter)
        return ''.join(letters)
    
    def apply_sequence_variations(self, vec, variations):
        name, max_error_length = vec
        if max_error_length > 0:
            num_kte_tse = len([var for var in variations if var in ['kte', 'tse']])
            if num_kte_tse>=1:
                max_error_length = max(1, max_error_length//num_kte_tse)
            for var in variations:                    
                name = self.variations_dict.get(var)(name, max_error_length)
        return name
        
    def generate_name_variations(self, dataset: pd_df, variations_name: str, frac: float=0.1, apply_col: str='name', orig_col: str=None, orig_variations_name:str=None, orig_error_length_col:str=None):
        # generate multiple sequential name variations using internal variation methods
        count = int(frac*len(dataset))
        variations = variations_name.split('_')
        np.random.seed(42)
        dataset = dataset.sample(n=count, random_state=42).reset_index(drop=True)
        dataset['max_error_length'] = self.generate_max_error_lengths(count)
        if orig_error_length_col is not None:
            dataset['max_error_length'] -= dataset[orig_error_length_col]
        dataset['name2'] = dataset[[apply_col, 'max_error_length']].progress_apply(self.apply_sequence_variations, axis=1, args=(variations,))
        if orig_col is None:
            orig_col = apply_col
        dataset = dataset.rename(columns={orig_col: 'name1'})
        if orig_variations_name is not None:
            variations_name = orig_variations_name + '_' + variations_name
        dataset['variations'] = variations_name
        dataset = dataset[['name1', 'name2', 'variations']]
        print(f'Done, Variations = {variations_name}')
        return dataset
        