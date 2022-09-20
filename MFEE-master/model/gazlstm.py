# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.crf import CRF
from model.layers import *
from transformers.modeling_bert import BertModel

class GazLSTM(nn.Module):
    def __init__(self, data):
        super(GazLSTM, self).__init__()

        self.gpu = data.HP_gpu
        self.use_biword = data.use_bigram
        self.use_gaz = data.use_gaz#改动--jie
        self.hidden_dim = data.HP_hidden_dim
        self.gaz_alphabet = data.gaz_alphabet
        self.gaz_emb_dim = data.gaz_emb_dim
        self.word_emb_dim = data.word_emb_dim
        self.biword_emb_dim = data.biword_emb_dim
        self.entity_emb_dim = 50#改动--jie
        self.use_char = data.HP_use_char
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.use_count = data.HP_use_count
        self.num_layer = data.HP_num_layer
        self.model_type = data.model_type
        self.use_bert = data.use_bert
        self.use_attention = True#改动--jie
        self.use_entity = True#改动--jie
        scale = np.sqrt(3.0 / self.gaz_emb_dim)
        data.pretrain_gaz_embedding[0,:] = np.random.uniform(-scale, scale, [1, self.gaz_emb_dim])

        if self.use_char:
            scale = np.sqrt(3.0 / self.word_emb_dim)
            data.pretrain_word_embedding[0,:] = np.random.uniform(-scale, scale, [1, self.word_emb_dim])
        self.gaz_embedding = nn.Embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.word_emb_dim)
        if self.use_biword:
            self.biword_embedding = nn.Embedding(data.biword_alphabet.size(), self.biword_emb_dim)

       
        if data.pretrain_gaz_embedding is not None:
            self.gaz_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        else:
            self.gaz_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)))
        #改动--jie
        print("final gaz_alphabet_size:",data.gaz_alphabet.size())#改动--jie

        if self.use_entity:#改动--jie
            self.entity_embedding = nn.Embedding(5,self.entity_emb_dim)#改动--jie
        # if data.pretrain_word_embedding is not None:#改动--jie
        #     self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        # else:
        #     self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.word_emb_dim)))
        if self.use_biword:
            if data.pretrain_biword_embedding is not None:
                self.biword_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_biword_embedding))
        #     else:
        #         self.biword_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.biword_alphabet.size(), self.word_emb_dim)))#改动--jie

        char_feature_dim = 0#改动--jie
        if self.use_bert:
            char_feature_dim += 768
        if self.use_biword:
            char_feature_dim += self.biword_emb_dim

        # 添加词典词汇信息
        if self.use_gaz:#改动--jie
            char_feature_dim += 4*self.gaz_emb_dim#改动--jie

        if self.use_entity:#改动--jie
            char_feature_dim += self.entity_emb_dim#改动--jie

        ## lstm model
        if self.model_type == 'lstm':
            lstm_hidden = self.hidden_dim
            if self.bilstm_flag:
                self.hidden_dim *= 2
            self.NERmodel = NERmodel(model_type='lstm', input_dim=char_feature_dim, hidden_dim=lstm_hidden, num_layer=self.lstm_layer, biflag=self.bilstm_flag)

        ## cnn model
        if self.model_type == 'cnn':
            self.NERmodel = NERmodel(model_type='cnn', input_dim=char_feature_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer, dropout=data.HP_dropout, gpu=self.gpu)

        ## attention model
        if self.model_type == 'transformer':
            self.NERmodel = NERmodel(model_type='transformer', input_dim=char_feature_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer, dropout=data.HP_dropout)

        self.drop = nn.Dropout(p=data.HP_dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, data.label_alphabet_size+2)
        self.crf = CRF(data.label_alphabet_size, self.gpu)

        if self.use_bert:
            self.bert_encoder = BertModel.from_pretrained('bert-base-chinese')#改动--jie
            for p in self.bert_encoder.parameters():
                p.requires_grad = False

        if self.gpu:
            self.gaz_embedding = self.gaz_embedding.cuda()
            # self.word_embedding = self.word_embedding.cuda()#改动--jie
            if self.use_biword:
                self.biword_embedding = self.biword_embedding.cuda()
            self.NERmodel = self.NERmodel.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()

            if self.use_bert:
                self.bert_encoder = self.bert_encoder.cuda()

            if self.use_entity:#改动--jie
                self.entity_embedding = self.entity_embedding.cuda()#改动--jie

        print('char_feature_dim:', char_feature_dim)#改动--jie

    def get_tags(self,gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask_input, gazchar_mask_input, mask, word_seq_lengths, batch_bert, bert_mask, ner_inputs):#改动--jie
        batch_size = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        max_gaz_num = layer_gaz.size(-1)
        gaz_match = []

        ### bert feature#改动--jie
        if self.use_bert:#改动--jie
            seg_id = torch.zeros(bert_mask.size()).long()
            if torch.cuda.is_available():
                seg_id = torch.zeros(bert_mask.size()).long().cuda()
            outputs = self.bert_encoder(batch_bert, bert_mask, seg_id)
            # 去除cls和sep
            outputs = outputs[0][:, 1:-1, :]
            word_input_cat = outputs
        #改动--jie
        #添加词嵌入
        # word_embs = self.word_embedding(word_inputs)#改动--jie
        # word_input_cat =torch.cat([word_input_cat,word_embs],dim=-1)

        #拼接biword
        if self.use_biword:
            biword_embs = self.biword_embedding(biword_inputs)
            word_input_cat = torch.cat([word_input_cat,biword_embs],dim=-1)


        #改动--jie
        # if self.use_char:
        #     gazchar_embeds = self.word_embedding(gaz_chars)
        #
        #     gazchar_mask = gazchar_mask_input.unsqueeze(-1).repeat(1,1,1,1,1,self.word_emb_dim)
        #     gazchar_embeds = gazchar_embeds.data.masked_fill_(gazchar_mask.data, 0)  #(b,l,4,gl,cl,ce)
        #
        #     # gazchar_mask_input:(b,l,4,gl,cl)
        #     gaz_charnum = (gazchar_mask_input == 0).sum(dim=-1, keepdim=True).float()  #(b,l,4,gl,1)
        #     gaz_charnum = gaz_charnum + (gaz_charnum == 0).float()
        #     gaz_embeds = gazchar_embeds.sum(-2) / gaz_charnum  #(b,l,4,gl,ce)
        #
        #     if self.model_type != 'transformer':
        #         gaz_embeds = self.drop(gaz_embeds)
        #     else:
        #         gaz_embeds = gaz_embeds
        #改动--jie
        # else:  #use gaz embedding
        if self.use_gaz:#改动--jie
            gaz_embeds = self.gaz_embedding(layer_gaz)#改动--jie

            if self.model_type != 'transformer':#改动--jie
                gaz_embeds_d = self.drop(gaz_embeds)#改动--jie
            else:#改动--jie
                gaz_embeds_d = gaz_embeds#改动--jie

            gaz_mask = gaz_mask_input.unsqueeze(-1).repeat(1,1,1,1,self.gaz_emb_dim)#改动--jie
            #对gaz_embeds进行初始化填充，有mask的地方用0代替
            gaz_embeds = gaz_embeds_d.data.masked_fill_(gaz_mask.bool().data, 0)  #(b,l,4,g,ge)  ge:gaz_embed_dim#改动--jie


            if self.use_count:
                count_sum = torch.sum(gaz_count, dim=3, keepdim=True)  #(b,l,4,gaz_num)
                count_sum = torch.sum(count_sum, dim=2, keepdim=True)  #(b,l,1,1)
                weights = gaz_count.div(count_sum)  #(b,l,4,gaz_num)
                weights = weights*4
                weights = weights.unsqueeze(-1)
                gaz_embeds = weights*gaz_embeds  #(b,l,4,gaz_num,embed_dim)
                gaz_embeds = torch.sum(gaz_embeds, dim=3)  #(b,l,4,embed_dim)
            else:
                gaz_num = (gaz_mask_input == 0).sum(dim=-1, keepdim=True).float()  #(b,l,4,1)
                gaz_embeds = gaz_embeds.sum(-2) / gaz_num  #(b,l,4,ge)/(b,l,4,1)

            gaz_embeds_cat = gaz_embeds.view(batch_size,seq_len,-1)  #(b,l,4*ge)

            # 拼接词典嵌入
            word_input_cat = torch.cat([word_input_cat, gaz_embeds_cat], dim=-1)  #(b,l,we+4*ge)

        #print(word_input_cat.size())

        # 拼接ner嵌入层#改动--jie
        if self.use_entity:
            entity_embeds = self.entity_embedding(ner_inputs)
            word_input_cat = torch.cat([word_input_cat,entity_embeds],dim=-1)

        feature_out_d = self.NERmodel(word_input_cat)

        tags = self.hidden2tag(feature_out_d)

        return tags, gaz_match      
        #改动--jie


    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask, batch_label, batch_bert, bert_mask,batch_ner):#改动--jie

        tags, _ = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count,gaz_chars, gaz_mask, gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask,batch_ner)#改动--jie

        total_loss = self.crf.neg_log_likelihood_loss(tags, mask.bool(), batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask.bool())

        return total_loss, tag_seq



    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,layer_gaz, gaz_count,gaz_chars, gaz_mask,gazchar_mask, mask, batch_bert, bert_mask,batch_ner):#改动--jie

        tags, gaz_match = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count,gaz_chars, gaz_mask, gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask,batch_ner)

        scores, tag_seq = self.crf._viterbi_decode(tags, mask.bool())

        return tag_seq, gaz_match


#########################################################################################################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################



class TypeCls(nn.Module):
    def __init__(self, config):
        super(TypeCls, self).__init__()
        self.type_emb = nn.Embedding(config.type_num, config.hidden_size)
        self.register_buffer('type_indices', torch.arange(0, config.type_num, 1).long())
        self.dropout = nn.Dropout(config.decoder_dropout)

        self.config = config
        self.Predictor = AdaptiveAdditionPredictor(config.hidden_size, dropout_rate=config.decoder_dropout)

    def forward(self, text_rep, mask):
        type_emb = self.type_emb(self.type_indices)
        pred = self.Predictor(type_emb, text_rep, mask)  # [b, c]
        p_type = torch.sigmoid(pred)
        return p_type, type_emb


class TriggerRec(nn.Module):
    def __init__(self, config, hidden_size):
        super(TriggerRec, self).__init__()
        self.ConditionIntegrator = ConditionalLayerNorm(hidden_size)
        self.SA = MultiHeadedAttention1(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)

        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.head_cls = nn.Linear(hidden_size, 1, bias=True)
        self.tail_cls = nn.Linear(hidden_size, 1, bias=True)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.config = config

    def forward(self, query_emb, text_emb, mask):
        '''

        :param query_emb: [b, e]
        :param text_emb: [b, t, e]
        :param mask: 0 if masked
        :return: [b, t, 1], [], []
        '''

        h_cln = self.ConditionIntegrator(text_emb, query_emb)

        h_cln = self.dropout(h_cln)
        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        h_sa = self.dropout(h_sa)
        inp = self.layer_norm(h_sa + h_cln)
        inp = gelu(self.hidden(inp))
        inp = self.dropout(inp)
        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, 1]
        p_e = torch.sigmoid(self.tail_cls(inp))  # [b, t, 1]
        return p_s, p_e, h_cln


class ArgsRec(nn.Module):
    def __init__(self, config, hidden_size, num_labels, seq_len, pos_emb_size):
        super(ArgsRec, self).__init__()
        self.relative_pos_embed = nn.Embedding(seq_len * 2, pos_emb_size)
        self.ConditionIntegrator = ConditionalLayerNorm(hidden_size)
        self.SA = MultiHeadedAttention1(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)
        self.hidden = nn.Linear(hidden_size + pos_emb_size, hidden_size)

        self.head_cls = nn.Linear(hidden_size, num_labels, bias=True)
        self.tail_cls = nn.Linear(hidden_size, num_labels, bias=True)

        self.gate_hidden = nn.Linear(hidden_size, hidden_size)
        self.gate_linear = nn.Linear(hidden_size, num_labels)

        self.seq_len = seq_len
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.config = config

    def forward(self, text_emb, relative_pos, trigger_mask, mask, type_emb):
        '''
        :param query_emb: [b, 4, e]
        :param text_emb: [b, t, e]
        :param relative_pos: [b, t, e]
        :param trigger_mask: [b, t]
        :param mask:
        :param type_emb: [b, e]
        :return:  [b, t, a], []
        '''
        trigger_emb = torch.bmm(trigger_mask.unsqueeze(1).float(), text_emb).squeeze(1)  # [b, e]
        trigger_emb = trigger_emb / 2

        h_cln = self.ConditionIntegrator(text_emb, trigger_emb)
        h_cln = self.dropout(h_cln)
        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        h_sa = self.dropout(h_sa)
        h_sa = self.layer_norm(h_sa + h_cln)

        rp_emb = self.relative_pos_embed(relative_pos)
        rp_emb = self.dropout(rp_emb)

        inp = torch.cat([h_sa, rp_emb], dim=-1)

        inp = gelu(self.hidden(inp))
        inp = self.dropout(inp)

        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, l]
        p_e = torch.sigmoid(self.tail_cls(inp))

        type_soft_constrain = torch.sigmoid(self.gate_linear(type_emb))  # [b, l]
        type_soft_constrain = type_soft_constrain.unsqueeze(1).expand_as(p_s)
        p_s = p_s * type_soft_constrain
        p_e = p_e * type_soft_constrain

        return p_s, p_e, type_soft_constrain


class CasEE(nn.Module):
    def __init__(self, config, model_weight, pos_emb_size):
        super(CasEE, self).__init__()
        self.bert = model_weight

        self.config = config
        self.args_num = config.args_num
        self.text_seq_len = config.seq_length

        self.type_cls = TypeCls(config)
        self.trigger_rec = TriggerRec(config, config.hidden_size)
        self.args_rec = ArgsRec(config, config.hidden_size, self.args_num, self.text_seq_len, pos_emb_size)
        self.dropout = nn.Dropout(config.decoder_dropout)

        self.loss_0 = nn.BCELoss(reduction='none')
        self.loss_1 = nn.BCELoss(reduction='none')
        self.loss_2 = nn.BCELoss(reduction='none')

    def forward(self, tokens, segment, mask, type_id, type_vec, trigger_s_vec, trigger_e_vec, relative_pos, trigger_mask, args_s_vec, args_e_vec, args_mask):
        '''

        :param tokens: [b, t]
        :param segment: [b, t]
        :param mask: [b, t], 0 if masked
        :param trigger_s: [b, t]
        :param trigger_e: [b, t]
        :param relative_pos:
        :param trigger_mask: [0000011000000]
        :param args_s: [b, l, t]
        :param args_e: [b, l, t]
        :param args_m: [b, k]
        :return:
        '''

        outputs = self.bert(
            tokens,
            attention_mask=mask,
            token_type_ids=segment,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )

        output_emb = outputs[0]
        p_type, type_emb = self.type_cls(output_emb, mask)
        p_type = p_type.pow(self.config.pow_0)
        type_loss = self.loss_0(p_type, type_vec)
        type_loss = torch.sum(type_loss)

        type_rep = type_emb[type_id, :]
        p_s, p_e, text_rep_type = self.trigger_rec(type_rep, output_emb, mask)
        p_s = p_s.pow(self.config.pow_1)
        p_e = p_e.pow(self.config.pow_1)
        p_s = p_s.squeeze(-1)
        p_e = p_e.squeeze(-1)
        trigger_loss_s = self.loss_1(p_s, trigger_s_vec)
        trigger_loss_e = self.loss_1(p_e, trigger_e_vec)
        mask_t = mask.float()  # [b, t]
        trigger_loss_s = torch.sum(trigger_loss_s.mul(mask_t))
        trigger_loss_e = torch.sum(trigger_loss_e.mul(mask_t))

        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, relative_pos, trigger_mask, mask, type_rep)
        p_s = p_s.pow(self.config.pow_2)
        p_e = p_e.pow(self.config.pow_2)
        args_loss_s = self.loss_2(p_s, args_s_vec.transpose(1, 2))  # [b, t, l]
        args_loss_e = self.loss_2(p_e, args_e_vec.transpose(1, 2))
        mask_a = mask.unsqueeze(-1).expand_as(args_loss_s).float()  # [b, t, l]
        args_loss_s = torch.sum(args_loss_s.mul(mask_a))
        args_loss_e = torch.sum(args_loss_e.mul(mask_a))

        trigger_loss = trigger_loss_s + trigger_loss_e
        args_loss = args_loss_s + args_loss_e

        type_loss = self.config.w1 * type_loss
        trigger_loss = self.config.w2 * trigger_loss
        args_loss = self.config.w3 * args_loss
        loss = type_loss + trigger_loss + args_loss
        return loss, type_loss, trigger_loss, args_loss

    def plm(self, tokens, segment, mask):
        assert tokens.size(0) == 1

        outputs = self.bert(
            tokens,
            attention_mask=mask,
            token_type_ids=segment,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )
        output_emb = outputs[0]
        return output_emb

    def predict_type(self, text_emb, mask):
        assert text_emb.size(0) == 1
        p_type, type_emb = self.type_cls(text_emb, mask)
        p_type = p_type.view(self.config.type_num).data.cpu().numpy()
        return p_type, type_emb

    def predict_trigger(self, type_rep, text_emb, mask):
        assert text_emb.size(0) == 1
        p_s, p_e, text_rep_type = self.trigger_rec(type_rep, text_emb, mask)
        p_s = p_s.squeeze(-1)  # [b, t]
        p_e = p_e.squeeze(-1)
        mask = mask.float()  # [1, t]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.text_seq_len).data.cpu().numpy()  # [b, t]
        p_e = p_e.view(self.text_seq_len).data.cpu().numpy()
        return p_s, p_e, text_rep_type

    def predict_args(self, text_rep_type, relative_pos, trigger_mask, mask, type_rep):
        assert text_rep_type.size(0) == 1
        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, relative_pos, trigger_mask, mask, type_rep)
        mask = mask.unsqueeze(-1).expand_as(p_s).float()  # [b, t, l]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.text_seq_len, self.args_num).data.cpu().numpy()
        p_e = p_e.view(self.text_seq_len, self.args_num).data.cpu().numpy()
        return p_s, p_e, type_soft_constrain
