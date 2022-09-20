with open('test.data.one.sentence.with.empty.sentence.txt','r',encoding='utf-8') as f:
    count = -1
    idx = []
    lines  = f.readlines()
    count_line = []
    sentence_count = []
    current_sentence = 0
    for id,line in enumerate(lines):
        if '\n' in line[0]:
            count = -1
            current_sentence+=1
        else:
            count+=1
        if 'B-' in line:
            idx.append(count)
            sentence_count.append(current_sentence)
            count_line.append(id)
print(idx)