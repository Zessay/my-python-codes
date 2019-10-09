from pathlib import Path
from collections import Counter
from pypinyin import Style, lazy_pinyin
import numpy as np 


class ChineseVector:
    def __init__(self, path="./", text="bishun_29685.txt"):
        self.fpath = Path(path) / text
        # 所有声母
        ## 严格来说，y和w不是声母，但是为了完整性；''表示没有声母
        self.initials = ['', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                       'g', 'k', 'h', 'j', 'q', 'x', 
                       'zh', 'ch', 'sh',
                       'r', 'z', 'c', 's',
                       'y', 'w']
        # 所有韵母
        self.finals = ['a', 'o', 'e', 'i',
                     'er', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 
                     'ang', 'eng', 'ong', 'ia', 'ie', 'iao', 'iou', 'ian', 'in', 'iang', 'ing', 'iong',
                     'u', 'ua', 'uo', 'uai', 'uei', 'uan', 'uen', 'uang', 'ueng', 
                     'v', 've', 'van', 'vn']
        # 记录声母和韵母的长度
        self.inLen = len(self.initials)
        self.finLen = len(self.finals)
        # 获取笔顺字典
        self.dictionary = self.load_bishun(self.fpath)
        
    def load_bishun(self, fpath, encoding='utf8'):
        '''
        加载字到笔顺码字典
        '''
        # 读取文件，按换行符分隔
        lines = fpath.read_text(encoding=encoding).split('\n')
        # 获取字典对应的笔顺码
        dictionary = {}
        for line in lines:
            # 以'\t'为分隔符进行分隔
            cols = line.split()
            # 取第二列和倒数第二列
            if cols:
                zi, bishun = cols[1], cols[-2]
                dictionary[zi] = bishun
        return dictionary
    
    def bihua_vec(self, zi):
        '''
        获取笔划向量，5维
        '''
        bishun = self.dictionary[zi]
        cnts = Counter(bishun)
        vec = np.zeros(5)
        for pos, num in cnts.items():
            vec[int(pos)-1] = num
        return vec.astype(np.int)
    
    def pinyin_vec(self, zi, onehot=True):
        # 首先是字的声母
        initial = lazy_pinyin(zi, style=Style.INITIALS, strict=False)[0]
        in_index = self.initials.index(initial)
        # 字的韵母和声调
        final = lazy_pinyin(zi, style=Style.FINALS_TONE3)[0]
        if final[-1] in '1234':
            shengdiao = final[-1]
            final = final[:-1]
        else:
            shengdiao = 0 
        fin_index = self.finals.index(final)
        # 将3种类别转换成onehot的形式
        if onehot:
            in_vec = self._onehot(self.inLen, int(in_index))
            fin_vec = self._onehot(self.finLen, int(fin_index))
            sh_vec = self._onehot(5, int(shengdiao))
            vec = np.hstack([in_vec, fin_vec, sh_vec])
            return vec.astype(np.int)
        else:
            return np.array([in_index, fin_index, shengdiao]).astype(np.int)
   
    def _onehot(self, length, index):
        '''
        给定长度和索引，转化为onehot
        '''
        vec = np.zeros(length)
        vec[index] = 1
        return vec
    
    def get_vec(self, zi, onehot=True):
        '''
        获取笔画和拼音组成的向量
        '''
        bihua_vec = self.bihua_vec(zi)
        pinyin_vec = self.pinyin_vec(zi, onehot=onehot)
        vec = np.hstack([bihua_vec, pinyin_vec]).astype(np.int)
        return vec


if __name__ == "__main__":
    basename = "../../data/utils/"
    ch = ChineseVector(basename)
    hanzi_vector = ch.get_vec("我", onehot=False)
    # print(hanzi_vector)  ## [ 2  1  2  1  1 23 25  3]