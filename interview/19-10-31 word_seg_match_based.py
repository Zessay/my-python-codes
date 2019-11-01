# 获取词典
def get_dic(wordDict_file):
    with open(test_file, 'r', encoding="utf8") as f:
        try:
            file_content = f.read().split()
        finally:
            f.close()
    chars = list(set(file_content))
    return chars


# 基于正向匹配的分词方法
def forward_matching(wordDict_file, test_file, tokens_file, max_len=5):
    h = open(tokens_file, 'w', encoding='utf8')
    with open(test_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    
    # 获取字典中的词
    dic = get_dic(wordDict_file)
    
    # 分别对每行进行正向最大匹配
    for line in lines:
        my_list = []
        ##表示该行字的数量
        len_hang = len(line)
        while len_hang>0:
            tryWord = line[0:max_len]
            while tryWord not in dic:
                if len(tryWord) == 1:
                    break
                
                tryWord = tryWord[0:len(tryWord)-1]
            my_list.append(tryWord)
            line = line[len(tryWord):]
            len_hang = len(line)
        
        # 将分词结果写入生成文件
        for t in my_list:
            if t == "\n":
                h.write("\n")
            else:
                h.write(t + " ")
    h.close()


# 基于逆向匹配的分词方法
def backword_matching(wordDict_file, test_file, tokens_file, max_len=5):
    h = open(test_file, 'w', encoding='utf8')
    with open(tokens_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    # 获取词典
    dic =  get_dic(wordDict_file)
    # 对测试文件中的每一行
    for line in lines:
        ## 定义一个栈
        my_stack = []
        ## 定义剩余词的长度为判断条件
        len_hang = len(line)
        ## 如果剩余词的长度为0则继续循环
        while len_hang > 0: 
            tryWord = line[-max_len:]
            ## 如果该词不在词表中，则继续往后匹配
            while tryWord not in dic:
                if len(tryWord) == 1:
                    break
                tryWord = tryWord[1:]
            my_stack.append(tryWord)
            ## 更新行的信息
            line = line[0:len(line)-len(tryWord)]
            ## 更新循环条件
            len_hang = len(line)
        
        while len(my_stack):
            t = my_stack.pop()
            if t == "\n":
                h.write("\n")
            else:
                h.write(t+" ")
    h.close()