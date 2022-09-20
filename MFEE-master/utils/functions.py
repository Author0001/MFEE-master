# -*- coding: utf-8 -*-

import sys
import numpy as np
import re
from utils.alphabet import Alphabet
from transformers.tokenization_bert import BertTokenizer
import jieba
import os
from LAC import LAC

NULLKEY = "-null-"
lac = LAC(mode='lac')
# LTP_DATA_DIR='D:/Python_tools/ltp_data'

# ltp_seg = pyltp.Segmentor()
# LTP_PATH = os.path.join(LTP_DATA_DIR,'cws.model')
# ltp_seg.load(LTP_PATH)

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance_with_gaz(num_layer, input_file, gaz, word_alphabet, biword_alphabet, biword_count, char_alphabet, gaz_alphabet, gaz_count, gaz_split, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    '''
    gaz[idx][label]:存放idx对应BMES的gaz_alphabet的index
    gazs_count[idx][label]：与gaz[idx][label]对应，从gaz_count对应索引获取次数
    gaz_char_Id[idex][label]:存放idx对应BMES的word_alphabet的所有字符
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    in_lines = open(input_file,'r',encoding="utf-8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_index = biword_alphabet.get_index(biword)
            biword_Ids.append(biword_index)
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words)>0):
                text = [''.join(word) for word in words]
                seg_rs,seg_idx,ner_rs = segment_wordlist(text)
                # for seg in seg_rs:
                #      gaz_alphabet.add(seg)
                #      index = gaz_alphabet.get_index(seg)
                #      gaz_count[index] = gaz_count.get(index, 0)  ## initialize gaz count
                gaz_Ids = []
                layergazmasks = []
                gazchar_masks = []
                w_length = len(words)
                gazs = [ [[] for i in range(4)] for _ in range(w_length)]  # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
                gazs_count = [ [[] for i in range(4)] for _ in range(w_length)]
                gaz_char_Id = [ [[] for i in range(4)] for _ in range(w_length)]  ## gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[[w1c1,w1c2,...],[],...]
                max_gazlist = 0
                max_gazcharlen = 0
                for idx in range(w_length):
                    matched_list = gaz.enumerateMatchList(words[idx:])
                    matched_length = [len(a) for a in matched_list]
                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    matched_count = [gaz_count[id] for  id in matched_Id]
                    max_count = max(matched_count) if len(matched_count)>0 else 1
                    #如果该text段有分词，就添加到matched_list中
                    if len(seg_idx) > 0 and idx == seg_idx[0]:
                        if seg_rs[0] not in matched_list:
                            matched_list.append(seg_rs[0])
                            matched_length.append(len(seg_rs[0]))
                            matched_Id.append(gaz_alphabet.get_index(seg_rs[0]))
                            #以当前序列中字符对应词汇信息的最高count赋予分词结果权重
                            matched_count.append(max_count)
                        else:
                            pos = matched_list.index(seg_rs[0])
                            matched_count[pos] += max_count
                        del (seg_idx[0])
                        del (seg_rs[0])
                    if matched_length:
                        max_gazcharlen = max(max(matched_length),max_gazcharlen)
                    for w in range(len(matched_Id)):
                        gaz_chars = []
                        g = matched_list[w]
                        for c in g:
                            gaz_chars.append(word_alphabet.get_index(c))
                        if matched_length[w] == 1:  ## Single
                            gazs[idx][3].append(matched_Id[w])
                            gazs_count[idx][3].append(1)
                            gaz_char_Id[idx][3].append(gaz_chars)
                        else:
                            gazs[idx][0].append(matched_Id[w])   ## Begin
                            gazs_count[idx][0].append(matched_count[w])
                            gaz_char_Id[idx][0].append(gaz_chars)
                            wlen = matched_length[w]
                            gazs[idx+wlen-1][2].append(matched_Id[w])  ## End
                            gazs_count[idx+wlen-1][2].append(matched_count[w])
                            gaz_char_Id[idx+wlen-1][2].append(gaz_chars)
                            for l in range(wlen-2):
                                gazs[idx+l+1][1].append(matched_Id[w])  ## Middle
                                gazs_count[idx+l+1][1].append(matched_count[w])
                                gaz_char_Id[idx+l+1][1].append(gaz_chars)
                    for label in range(4):
                        if not gazs[idx][label]:
                            gazs[idx][label].append(0)
                            gazs_count[idx][label].append(1)
                            gaz_char_Id[idx][label].append([0])
                        max_gazlist = max(len(gazs[idx][label]),max_gazlist)
                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]  #词号
                    if matched_Id:
                        #[[matched_Id列表],[对应的长度列表]] 如[[9487,426][2,1]]表示9487的长度为2，426长度为1
                        gaz_Ids.append([matched_Id, matched_length])
                    else:
                        gaz_Ids.append([])
                ## batch_size = 1
                for idx in range(w_length):
                    gazmask = []
                    gazcharmask = []
                    #0表示不mask，1表示需要mask
                    for label in range(4):
                        label_len = len(gazs[idx][label])
                        count_set = set(gazs_count[idx][label])
                        if len(count_set) == 1 and 0 in count_set:
                            gazs_count[idx][label] = [1]*label_len
                        mask = label_len*[0]
                        mask += (max_gazlist-label_len)*[1]
                        gazs[idx][label] += (max_gazlist-label_len)*[0]  ## padding
                        gazs_count[idx][label] += (max_gazlist-label_len)*[0]  ## padding
                        char_mask = []
                        for g in range(len(gaz_char_Id[idx][label])):
                            glen = len(gaz_char_Id[idx][label][g])
                            charmask = glen*[0]
                            charmask += (max_gazcharlen-glen) * [1]
                            char_mask.append(charmask)
                            gaz_char_Id[idx][label][g] += (max_gazcharlen-glen) * [0]
                        gaz_char_Id[idx][label] += (max_gazlist-label_len)*[[0 for i in range(max_gazcharlen)]]
                        char_mask += (max_gazlist-label_len)*[[1 for i in range(max_gazcharlen)]]
                        gazmask.append(mask)
                        gazcharmask.append(char_mask)
                    layergazmasks.append(gazmask)
                    gazchar_masks.append(gazcharmask)
                texts = ['[CLS]'] + words + ['[SEP]']
                bert_text_ids = tokenizer.convert_tokens_to_ids(texts)
                instence_texts.append([words, biwords, chars, gazs, labels])
                instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids, gazs, gazs_count, gaz_char_Id, layergazmasks,gazchar_masks, bert_text_ids,ner_rs])
            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []
    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    pretrain_emb[0,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
    for word, index in word_alphabet.instance2index.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r',encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                continue
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim



# def get_segment_result(word_list):
#     '''
#         返回两个分词工具分词结果的交集
#     '''
#     sentence = ''.join(word_list)
#     jb_rs = jieba.cut(sentence, cut_all=True)
#     ltp_rs = ltp_seg.segment(sentence)
#     jb_set = set(jb_rs)
#     ltp_set = set(ltp_rs)
#     seg_list = list(jb_set.intersection(ltp_set))
#     seg_list.sort(key=list(ltp_rs).index)
#     return seg_list



def get_segresult_for_idx(seg_list, word_segment):
    for seg in seg_list:
        if seg.startswith(word_segment[0]):
            return seg
    return None


def segment_wordlist(wordlist):
    # reg_rs = get_segment_result(wordlist)
    text = ''.join(wordlist)
    reg_rst = lac.run(text)
    reg_rs = reg_rst[0]
    ner_rs = reg_rst[1]
    reg_idx = []
    idx = 0
    for i in range(len(reg_rs)):
        while idx < len(text):
            if text[idx:].startswith(reg_rs[i]):
              reg_idx.append(idx)
              idx+=len(reg_rs[i])
              break
            idx+=1
    for i in range(len(ner_rs)):
        if ner_rs[i] == 'PER':
            ner_rs[i] = 1
        elif ner_rs[i] == 'TIME':
            ner_rs[i] = 2
        elif ner_rs[i] == 'ORG':
            ner_rs[i] = 3
        elif ner_rs[i] == 'LOC':
            ner_rs[i] = 4
        else:
            ner_rs[i] = 0
    ner_rst = []
    for i in range(len(reg_idx)):
        st_idx = reg_idx[i]
        end_idx = reg_idx[i+1] if i+1 <len(reg_idx) else len(text)
        for j in range(st_idx,end_idx):
            ner_rst.append(ner_rs[i])
    return reg_rs,reg_idx,ner_rst

if __name__=='__main__':
    wordlist=['再', '过', '两', '个', '礼', '拜', '0', '0', '世', '纪', '就', '要', '到', '了', '，', '您', '知', '道', '今', '年', '哪', '些', '人',
     '是', '国', '内', '大', '学', '生', '心', '目', '中', '的', '风', '云', '人', '物', '吗', '？']
    reg_rs,reg_idx,ner_rs = segment_wordlist(wordlist)
    print(reg_rs)
    print(reg_idx)
    print(ner_rs)
    print(len(wordlist))
    print(len(ner_rs))