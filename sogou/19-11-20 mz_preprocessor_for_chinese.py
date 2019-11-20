import pandas as pd 
import matchzoo as mz
import numpy as np 
import torch
import torch.nn as nn

from matchzoo.preprocessors.units import Unit
import jieba

# 去除停止词的类
class CNStopRemoval(Unit):
    def __init__(self, stopwords: list = []):
        self._stop = stopwords

    def transform(self, input_: list) -> list:
        return [token for token in input_ if token not in self._stop]

    @property
    def stopwords(self) -> list:
        """
        getter of stopwords
        :return:
        """
        return self._stop

# 去除标点的类
class CNPuncRemoval(Unit):
    _CNPuncs = ['。', '，', '！', '？', '、', '；', '：', '“',
                '”', '‘', '’', '（', '）', '【', '】', '{', '}',
                '『', '』', '「', '」', '〔', '〕', '——', '……', '—', '-',
                '～', '·', '《', '》', '〈', '〉', '﹏', '___', '.']
    _ENPuncs = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/',
              '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_',
              '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×',
              '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►',
              '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
              '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨',
              '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
              '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣',
              '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

    def transform(self, input_: list) -> list:
        return [token for token in input_
                if token not in self._CNPuncs+self._ENPuncs]


# 分词的类

class CNTokenize(Unit):
    """Process unit for text tokenization."""

    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """

        return jieba.lcut(input_)

class CNCharTokenize(Unit):
    """基于字符进行分隔"""
    def transform(self, input_: str) -> list:
        result = ""
        for ch in input_:
            if is_chinese_char(ch):
                result += ' ' + ch + ' '
            else:
                result += ch

        return result.split()


def is_chinese_char(ch):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    cp = ord(ch)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


# -----------------------------------------------------------------------------------------------------------------

from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from matchzoo.preprocessors import units
from matchzoo.preprocessors.build_vocab_unit import build_vocab_unit
from matchzoo.preprocessors.build_unit_from_data_pack import build_unit_from_data_pack
from matchzoo.preprocessors.chain_transform import chain_transform

class ChinesePreprocessor(BasePreprocessor):
    def __init__(self,
                 tokenize_mode: str = 'word',
                 truncated_mode: str = 'post',
                 truncated_length_left: int = None,
                 truncated_length_right: int = None,
                 filter_mode: str = 'df',
                 filter_low_freq: float = 1,
                 filter_high_freq: float = float('inf'),
                 stopwords: list = [],
                 remove_punc: bool = False,
                 lowercase: bool = False):
        """
        Preprocessor for Chinese
        :param tokenize_mode: 'word'表示分词，'char'表示分字
        """
        super().__init__()
        self._units = []
        self._truncated_mode = truncated_mode
        self._truncated_length_left = truncated_length_left
        self._truncated_length_right = truncated_length_right
        if self._truncated_length_left:
            self._left_truncatedlength_unit = units.TruncatedLength(
                self._truncated_length_left, self._truncated_mode
            )
        if self._truncated_length_right:
            self._right_truncatedlength_unit = units.TruncatedLength(
                self._truncated_length_right, self._truncated_mode
            )

        self._filter_unit = units.FrequencyFilter(
            low=filter_low_freq,
            high=filter_high_freq,
            mode=filter_mode
        )
        ## 定义分词的方式
        self._tokenize_unit = None
        if tokenize_mode == 'word':
            self._tokenize_unit = CNTokenize()
        elif tokenize_mode == 'char':
            self._tokenize_unit = CNCharTokenize()
        else:
            raise ValueError(f"This tokenize mode {tokenize_mode} is not defined.")
        self._units.append(self._tokenize_unit)

        ## 是否去除停止词
        if stopwords:
            self._units.append(CNStopRemoval(stopwords))
        ## 是否去除标点
        if remove_punc:
            self._units.append(CNPuncRemoval())
        ## 是否将英文字母小写
        if lowercase:
            self._units.append(units.lowercase.Lowercase())


    def fit(self, data_pack: DataPack, verbose: int=1):
        ## 经过分词、去标点以及去停用词
        data_pack = data_pack.apply_on_text(chain_transform(self._units),
                                            verbose=verbose)
        ## 过滤高频词和低频词
        ## 先通过build进行统计
        fitted_filter_unit = build_unit_from_data_pack(self._filter_unit,
                                                       data_pack,
                                                       flatten=False,
                                                       mode='right',
                                                       verbose=verbose)
        ### 基于上面统计的结果进行转换并保存模型
        data_pack = data_pack.apply_on_text(fitted_filter_unit.transform,
                                            mode='right', verbose=verbose)
        self._context['filter_unit'] = fitted_filter_unit

        ## 构建词表
        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)
        self._context['vocab_unit'] = vocab_unit

        vocab_size = len(vocab_unit.state['term_index'])
        self._context['vocab_size'] = vocab_size
        self._context['embedding_input_dim'] = vocab_size

        return self

    def transform(self, data_pack: DataPack, verbose: int = 1 ) -> DataPack:
        data_pack = data_pack.copy()
        data_pack.apply_on_text(chain_transform(self._units), verbose=verbose)
        data_pack.apply_on_text(self._context['filter_unit'].transform,
                                mode='right', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._context['vocab_unit'].transform,
                                mode='both', inplace=True, verbose=verbose)
        if self._truncated_length_left:
            data_pack.apply_on_text(self._left_truncatedlength_unit.transform,
                                    mode='left', inplace=True, verbose=verbose)
        if self._truncated_length_right:
            data_pack.apply_on_text(self._right_truncatedlength_unit.transform,
                                    mode='right', inplace=True, verbose=verbose)
        data_pack.append_text_length(inplace=True, verbose=verbose)
        data_pack.drop_empty(inplace=True)
        return data_pack
