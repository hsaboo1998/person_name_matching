import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import re
from unidecode import unidecode
from itertools import permutations

from pyphonetics import RefinedSoundex
from phonetics import dmetaphone
from textdistance import Levenshtein, DamerauLevenshtein as DL, JaroWinkler as JW, SmithWaterman as SW, NeedlemanWunsch as NW
from textdistance import Jaccard, Sorensen, Overlap, Bag
from textdistance import Editex
from textdistance import LCSSeq, LCSStr
from textdistance import BZ2NCD, ZLIBNCD

class Preprocessor:
    titles = ['mr', 'mrs', 'miss', 'ms', 'dr', 'prof', 'sir', 'rev', 'lady', 'lord', 'sheikh', 'sayyid', 'sayyida',\
              'abu', 'um', 'al', 'sri', 'smt', 'pandit', 'swami', 'baba', 'sensei', 'san', 'sama', 'kun', 'chan', 'shi',\
              'sifu', 'chief', 'baba', 'don', 'doctor', 'padre', 'lic', 'monsieur', 'madame', 'mademoiselle', 'herr', 'frau']
    @staticmethod
    def get_permutations(name):
        name_permutations = sorted([' '.join(parts_of_name) for parts_of_name in set(permutations(name.split()))])
        np.random.seed(42)
        np.random.shuffle(name_permutations)
        return name_permutations
    
    @classmethod
    def get_alignments(cls, names):
        name1_permutations = cls.get_permutations(names[0])
        name2_permutations = cls.get_permutations(names[1])
        alignment_budget = 10
        n_alignments_name1 = int(np.ceil(alignment_budget*len(name1_permutations)/(len(name1_permutations)+len(name2_permutations))))
        n_alignments_name2 = int(np.ceil(alignment_budget*len(name2_permutations)/(len(name1_permutations)+len(name2_permutations))))
        return name1_permutations[0:n_alignments_name1], name2_permutations[0:n_alignments_name2]
    
    @classmethod
    def get_best_alignment(cls, names):
        alignments_name1, alignments_name2 = cls.get_alignments(names)
        name_pairs = list(zip(*[arr.flatten() for arr in np.meshgrid(alignments_name1, alignments_name2)]))
        name_pairs_sorted = sorted(name_pairs, key=lambda vec: cls.needleman_wunsch(vec[0], vec[1]), reverse=True)
        return name_pairs_sorted[0]
    
    @classmethod
    def preprocess_names(cls, names):
        preprocessed_names = []
        for name in names:
            name = str(name).lower().strip()
            name = ' '.join(name.split())
            name = unidecode(name)
            name = re.sub('[^a-z ]', '', name)
            name = ' '.join([part for part in name.split() if part not in cls.titles])
            preprocessed_names.append(name)
        preprocessed_names = cls.get_best_alignment(preprocessed_names)
        return preprocessed_names
    
class FeatureGenerator:
    @staticmethod
    def pad_name(name, qval):
        name = ''.join((['<']*(qval-1))) + name + ''.join((['>']*(qval-1)))
        return name
    @classmethod
    def get_similarity(cls, names, sim_func, **kwargs):
        if 'qval' in kwargs:
            names = [cls.pad_name(name, kwargs.get('qval', 1)) for name in names]
        return sim_func(*names, **kwargs)
    @staticmethod
    def soundex(name1, name2):
        rs = RefinedSoundex()
        return 1 - rs.distance(name1, name2, metric='levenshtein')/max(len(name1), len(name2))
    @staticmethod
    def double_metaphone(name1, name2):
        p1 = set([p for p in dmetaphone(name1) if p!=''])
        p2 = set([p for p in dmetaphone(name2) if p!=''])
        return len(p1.intersection(p2))/len(p1.union(p2))
    @staticmethod
    def jaccard(name1, name2, qval):
        jc = Jaccard(qval)
        return jc(name1, name2)
    @staticmethod
    def sorenson(name1, name2, qval):
        sn = Sorensen(qval)
        return sn(name1, name2)
    @staticmethod
    def overlap(name1, name2, qval):
        ov = Overlap(qval)
        return ov(name1, name2)
    @staticmethod
    def bag(name1, name2):
        bag = Bag()
        return 1 - bag(name1, name2)/max(len(name1), len(name2))
    @staticmethod
    def levenshtein(name1, name2):
        lev = Levenshtein()
        return 1 - lev(name1, name2)/max(len(name1), len(name2))
    @staticmethod
    def dlevenshtein(name1, name2):
        dl = DL()
        return 1 - dl(name1, name2)/max(len(name1), len(name2))
    @staticmethod
    def smith_waterman(name1, name2):
        sw = SW()
        return sw(name1, name2)/max(len(name1), len(name2))
    @staticmethod
    def needleman_wunsch(name1, name2):
        nw = NW()
        return nw(name1, name2)/max(len(name1), len(name2))
    @staticmethod
    def jaro_winkler(name1, name2):
        jw = JW()
        return jw(name1, name2)
    @staticmethod
    def editex(name1, name2):
        ed = Editex()
        return 1 - ed(name1, name2)/(2*max(len(name1), len(name2)))
    @staticmethod
    def lcsseq(name1, name2):
        lsq = LCSSeq()
        return len(lsq(name1, name2))/max(len(name1), len(name2))
    @staticmethod
    def lcsstr(name1, name2):
        lsr = LCSStr()
        return len(lsr(name1, name2))/max(len(name1), len(name2))
    @staticmethod
    def bz2ncd(name1, name2):
        bzn = BZ2NCD()
        return 1 - bzn(name1, name2)
    @staticmethod
    def zlibncd(name1, name2):
        zzn = ZLIBNCD()
        return 1 - zzn(name1, name2)
    @classmethod
    def generate_features(cls, df):
        df['soundex'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.soundex,), axis=1)
        df['double_metaphone'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.double_metaphone,), axis=1)
        df['jaccard_2'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.jaccard,), qval=2, axis=1)
        df['jaccard_3'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.jaccard,), qval=3, axis=1)
        df['sorenson_2'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.sorenson,), qval=2, axis=1)
        df['sorenson_3'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.sorenson,), qval=3, axis=1)
        df['overlap_2'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.overlap,), qval=2, axis=1)
        df['overlap_3'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.overlap,), qval=3, axis=1)
        df['bag'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.bag,), axis=1)
        df['levenshtein'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.levenshtein,), axis=1)
        df['dlevenshtein'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.dlevenshtein,), axis=1)
        df['jaro_winkler'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.jaro_winkler,), axis=1)
        df['smith_waterman'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.smith_waterman,), axis=1)
        df['editex'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.editex,), axis=1)
        df['lcsseq'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.lcsseq,), axis=1)
        df['lcsstr'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.lcsstr,), axis=1)
        df['bz2ncd'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.bz2ncd,), axis=1)
        df['zlibncd'] = df[['name1', 'name2']].progress_apply(cls.get_similarity, args=(cls.zlibncd,), axis=1)
        return df
    
class NameMatcher(Preprocessor, FeatureGenerator):
    def __init__(self, model, selected_features, threshold):
        self.model = model
        self.threshold = threshold
        self.selected_features = selected_features
    
    @staticmethod
    def is_control_mechanism(names):
        parts_of_name1 = set(names[0].split())
        parts_of_name2 = set(names[1].split())
        num_common_parts = sum([(p1 in p2)|(p2 in p1) for p1 in parts_of_name1 for p2 in parts_of_name2])
        if num_common_parts in [len(parts_of_name1), len(parts_of_name2)]:
            return 1
        return 0
    
    def predict_using_pandas_dataframe(self, df: pd.DataFrame, target_name_col:str='name1', flagged_name_col:str='name2', require_proba=False, apply_control_mechanism=True) -> list:
        df = df[[target_name_col, flagged_name_col]]
        df = df.rename(columns={target_name_col: 'name1', flagged_name_col: 'name2'})
        print('Preprocessing started.')
        df[['name1', 'name2']] = df[['name1', 'name2']].progress_apply(lambda vec: self.preprocess_names(vec), axis=1, result_type='expand')
        print('Preprocessing completed. Feature Generation in progress...')
        df = self.generate_features(df)
        print('Feature generation completed.')
        df['proba'] = self.model.predict_proba(df[self.selected_features])[:,1]
        df['result'] = (df['proba']>self.threshold).astype(int)
        if apply_control_mechanism:
            df['cm'] = df[['name1', 'name2']].apply(lambda vec: self.is_control_mechanism(vec), axis=1)
            df['result'] = ((df['result']+df['cm'])>0).astype(int)
        if require_proba:
            df['result'] = list(zip(df['result'].tolist(), df['proba'].tolist()))
        return df['result'].tolist()

    def predict_using_names(self, target_name, flagged_name, require_proba=False, apply_control_mechanism=True):
        preprocessed_names = self.preprocess_names([target_name, flagged_name])
        df = pd.DataFrame([preprocessed_names], columns=['name1', 'name2'])
        result = self.predict_using_pandas_dataframe(df, require_proba=require_proba, apply_control_mechanism=apply_control_mechanism)
        return result[0]

