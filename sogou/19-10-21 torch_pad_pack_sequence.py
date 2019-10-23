import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def pad_sequence(vector_seqs, embeded):
    '''
    vector_seqs是数值化之后的输入，形状是[batch, *]
    里面各个句子的长度可以是不一致的
    
    embeded是nn.Embedding对象
    '''
    # 获取各个序列的实际长度
    seq_lengths = torch.LongTensor([len(seq) for seq in vector_seqs]).to(device)
    # 对序列进行左对齐，不满足最大长度的在末尾补0
    seq_tensor = torch.zeros((len(vector_seqs), seq_lengths.max())).long().to(device)
    for idx, (seq, seqlen) in enumerate(zip(vector_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    
    # 将seq_lengths按照长度递减的顺序排序
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    
    # 如果设置参数batch_first=True，那么输入的张量形状为[B,L,D]
    # 否则，输入的张量形状为 [L, B, D]
    seq_tensor = seq_tensor.transpose(0, 1)
    # 对序列进行编码，必须先获取词向量
    seq_tensor = embeded(seq_tensor)
    # 封装输入的张量
    packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())
    
    return packed_input, perm_idx


def unpad_sequence(packed_output, perm_idx):
    '''
    packed_output是输出，perm_idx是重排序之后的每个位置的原索引
    '''
    output, _ = pad_packed_sequence(packed_output)
    # 还原回batch_first
    output = output.transpose(0, 1)
    
    # 获取由最终得到的序列还原之前序列顺序的张量
    _, unperm_idx = perm_idx.sort(0)
    output = output[unperm_idx]
    
    return output



def test():
    seqs = ["gigantic_string", "tiny_str", "medium_str"]
    # 获取词表，最好设置<pad>的索引为0
    vocab = ['<pad>'] + sorted(set(''.join(seqs)))
    # 构造模型
    embeded = nn.Embedding(len(vocab), 10).to(device)
    lstm = nn.LSTM(10, 5).to(device)

    # 对序列进行向量化
    vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]
    packed_input, perm_idx = pad_sequence(vectorized_seqs, embeded)
    
    # 将封装之后的张量输入到网络中
    packed_output, (ht, ct) = lstm(packed_input)
    output = unpad_sequence(packed_output, perm_idx)
    print(output)