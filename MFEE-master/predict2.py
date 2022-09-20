import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle
import os
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.gazlstm import GazLSTM as SeqModel
from utils.data import Data
from main import batchify_with_label
import matplotlib.pyplot as plt


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(int(pred_tag[idx][idy])) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]

        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)

    return pred_label, gold_label

def set_seed(seed_num=1023):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)


def predict(data, model,output_file):
    file = output_file
    instances = data.raw_Ids
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    gazes = []
    record = 0
    for batch_id in range(total_batch):
        record += 1
        with torch.no_grad():
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = instances[start:end]
            if not instance:
                continue
            gaz_list, batch_word, batch_biword, batch_wordlen, batch_label, layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask, batch_bert, bert_mask, batch_ner = batchify_with_label(
                instance, data.HP_gpu, data.HP_num_layer, True)
            tag_seq, gaz_match = model(gaz_list, batch_word, batch_biword, batch_wordlen, layer_gaz, gaz_count,
                                       gaz_chars, gaz_mask, gazchar_mask, mask, batch_bert, bert_mask, batch_ner)
            gaz_list = [data.gaz_alphabet.get_instance(id) for batchlist in gaz_match if len(batchlist) > 0 for id in
                        batchlist]
            gazes.append(gaz_list)
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet)
            pred_results += pred_label
            gold_results += gold_label
    decode_time = time.time() - start_time
    try:
        speed = len(instances) / decode_time
    except:
        print(raw_file)
    # acc, iP, iR, iF, P, R, F = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    with open(file,'w',encoding='utf-8') as f:
        for i in range(len(data.raw_texts)):
            # for j in range(len(data.raw_texts[i][0])):
            #     f.write(data.raw_texts[i][0][j]+' '+pred_results[i][j]+'\n')
            for j in range(len(pred_results[i])):
                if j > 0 and pred_results[i][j].startswith('I-') and pred_results[i][j-1].startswith('O'):
                    pred_results[i][j]='B'+pred_results[i][j][1:]
                # if j >0 and j < len(pred_results[i]) and pred_results[i][j].startswith('O') and pred_results[i][j-1].startswith('I-') and pred_results[i][j+1].startswith('I-'):
                #     pred_results[i][j] =
            result = ','.join(pred_results[i])
            f.write(result)
            f.write('\n')

def load_model_decode(model_dir, data, name, gpu, seg=True):
    data.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
    model = SeqModel(data)

    model.load_state_dict(torch.load(model_dir))

    # print(("Decode %s data ..." % (name)))
    # start_time = time.time()
    # speed, acc, iP, iR, iF, p, r, f, pred_results, gazs = evaluate(data, model, name)
    # end_time = time.time()
    # time_cost = end_time - start_time
    # if seg:
    #     print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
    #     name, time_cost, speed, acc, p, r, f)))
    # else:
    #     print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc)))
    return model


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', help='Embedding for words', default='None')
    parser.add_argument('--status', choices=['train', 'test'], help='update algorithm', default='train')
    parser.add_argument('--modelpath', default="save_model/")
    parser.add_argument('--modelname', default="model")
    parser.add_argument('--savedset', help='Dir of saved data setting', default="data/save.dset")
    parser.add_argument('--seg', default="True")
    parser.add_argument('--extendalphabet', default="True")
    parser.add_argument('--raw',default='F:/data/zrzyb/trigger_input')
    parser.add_argument('--output',default='F:/data/zrzyb/trigger_output')
    parser.add_argument('--seed', default=965, type=int) #best:965
    parser.add_argument('--labelcomment', default="")
    parser.add_argument('--resultfile', default="result/result.txt")
    parser.add_argument('--num_iter', default=50, type=int)
    parser.add_argument('--num_layer', default=4, type=int)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=250)
    parser.add_argument('--model_type', default='lstm')
    parser.add_argument('--drop', type=float, default=0.6 )
    parser.add_argument('--use_biword', dest='use_biword', action='store_true', default=False)
    parser.add_argument('--use_gaz', dest='use_gaz', action='store_true', default=True)
    # parser.set_defaults(use_biword=False)
    parser.add_argument('--use_char', dest='use_char', action='store_true', default=False)
    # parser.set_defaults(use_biword=False)
    parser.add_argument('--use_count', action='store_true', default=True)
    parser.add_argument('--use_bert', action='store_true', default=True)
    args = parser.parse_args()
    seed_num = args.seed
    set_seed(seed_num)

    raw_path = args.raw
    output_dir = args.output
    # model_dir = args.loadmodel

    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = args.modelpath + args.modelname
    save_data_name = args.savedset
    gpu = torch.cuda.is_available()

    gaz_file = "../CNNNERmodel/data/ctb.50d.vec"
    char_emb = None
    bichar_emb = None
    # gaz_file=None
    sys.stdout.flush()
    with open(save_data_name, 'rb') as fp:
        data = pickle.load(fp)
    data.HP_num_layer = args.num_layer
    data.HP_batch_size = args.batch_size
    data.HP_iteration = args.num_iter
    data.label_comment = args.labelcomment
    data.result_file = args.resultfile
    data.HP_lr = args.lr
    data.use_bigram = args.use_biword
    data.HP_use_char = args.use_char
    data.HP_hidden_dim = args.hidden_dim
    data.HP_dropout = args.drop
    data.HP_use_count = args.use_count
    data.model_type = args.model_type
    data.use_bert = args.use_bert
    data.show_data_summary()

    files = os.listdir(raw_path)
    model = SeqModel(data)
    model.load_state_dict(torch.load(save_model_dir))
    for raw_file in files:
        input_file = os.path.join(raw_path,raw_file)
        output_file = os.path.join(output_dir,raw_file)
        data.raw_Ids = []
        # data.build_alphabet(input_file)
        data.generate_instance_with_gaz(input_file, 'raw')
        predict(data,model,output_file)
