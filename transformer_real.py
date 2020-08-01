import numpy as np
import random
import cv2
from matplotlib import pyplot as plt
from scipy import stats
import glob
import pickle as p
from sklearn.svm import SVC
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torchvision.models as models
from torch.autograd import Variable
import string
import tensorflow as tf

def cleanup(lines_raw):
    lines=[]
    for i in lines_raw:
        s='<start> '
        for x in i:
            if (x not in string.punctuation) and x!='\n':
                s+=x
        lines.append(s.lower()+' <end>')
        #lines.append(s.lower())
    return lines

f = open('data/train.en','r',encoding="utf-8")
eng_lines_raw = f.readlines()
f.close()

f = open('data/train.hi','r',encoding="utf-8")
hin_lines_raw = f.readlines()
f.close()

f = open('data/dev.en','r',encoding="utf-8")
eng_lines_raw_dev = f.readlines()
f.close()

f = open('data/dev.hi','r',encoding="utf-8")
hin_lines_raw_dev = f.readlines()
f.close()

eng_lines=cleanup(eng_lines_raw)
hin_lines=cleanup(hin_lines_raw)

eng_lines_dev=cleanup(eng_lines_raw_dev)
hin_lines_dev=cleanup(hin_lines_raw_dev)

eng_lines = eng_lines+eng_lines_dev
hin_lines = hin_lines+hin_lines_dev

#english
tokenizer_english = tf.keras.preprocessing.text.Tokenizer(num_words=5000,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer_english.fit_on_texts(eng_lines)

tokenizer_english.word_index['<pad>'] = 0
tokenizer_english.index_word[0] = '<pad>'

eng_lines = tokenizer_english.texts_to_sequences(eng_lines)
eng_lines = tf.keras.preprocessing.sequence.pad_sequences(eng_lines, padding='post')



#hindi
tokenizer_hindi = tf.keras.preprocessing.text.Tokenizer(num_words=5000,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer_hindi.fit_on_texts(hin_lines)

tokenizer_hindi.word_index['<pad>'] = 0
tokenizer_hindi.index_word[0] = '<pad>'

hin_lines = tokenizer_hindi.texts_to_sequences(hin_lines)
hin_lines = tf.keras.preprocessing.sequence.pad_sequences(hin_lines, padding='post')

eng_maxlen = len(eng_lines[0])
hin_maxlen = len(hin_lines[0])
vocab_len=5001
print(hin_maxlen,eng_maxlen)

eng_lines_train = eng_lines[:int(.8*len(eng_lines))]
eng_lines_dev = eng_lines[int(.8*len(eng_lines)):]
hin_lines_train = hin_lines[:int(.8*len(hin_lines))]
hin_lines_dev = hin_lines[int(.8*len(hin_lines)):]


device = 'cuda'

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        embed_size = 300
        self.src_word_embedding = nn.Embedding(vocab_len, embed_size)
        self.src_position_embedding = nn.Embedding(eng_maxlen, embed_size)
        self.trg_word_embedding = nn.Embedding(vocab_len, embed_size)
        self.trg_position_embedding = nn.Embedding(hin_maxlen, embed_size)

        self.transformer = nn.Transformer(embed_size,4,2,2,4,.1)

        self.out = nn.Linear(embed_size, vocab_len)
        self.dropout = nn.Dropout(.1)

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == 0 #<pad> has idx 0
        return src_mask.to(device)

    def forward(self, src, trg):

        B = src.shape[1]

        src_positions = torch.arange(0, src.shape[0]).unsqueeze(1).expand(src.shape[0], B).to(device)
        trg_positions = torch.arange(0, trg.shape[0]).unsqueeze(1).expand(trg.shape[0], B).to(device)


        embed_src = self.dropout((self.src_word_embedding(src) + self.src_position_embedding(src_positions)))
        embed_trg = self.dropout((self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)))

        
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.shape[0]).to(device)
        src_padding_mask = self.make_src_mask(src)

        #print(embed_src.shape,embed_trg.shape,src_padding_mask.shape,trg_mask.shape)
        op = self.transformer(embed_src,embed_trg,src_key_padding_mask=src_padding_mask,tgt_mask=trg_mask)

        #print(op)
        
        op = self.out(op)
        #op = torch.exp(op)
        #op = op/(torch.sum(op,dim=2)).unsqueeze(1)
        #op = nn.Sigmoid()(op)
        #print(op.shape)
        #op = op/(torch.sum(op,dim=2)).unsqueeze(1)
        #print(op.shape)
        #print(op)
        return op


def translate_sentence(model, sentence, max_length=50):

    sentence_tensor = torch.LongTensor(sentence).unsqueeze(1).to(device)

    #outputs = [english.vocab.stoi["<sos>"]]
    outputs = [tokenizer_hindi.word_index['<start>']]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :]
        #print(best_guess)
        best_guess = best_guess.item()
        outputs.append(best_guess)

        if best_guess == tokenizer_hindi.word_index['<end>']:
            break

    translated_sentence = [tokenizer_hindi.index_word[idx] for idx in outputs]
    return translated_sentence



model = Transformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

criterion = nn.CrossEntropyLoss(ignore_index=0)


print("training")

TRAIN=0
b=32 #batch size

subsample = 30000

f=open("tans_dump.txt",'wb')
if TRAIN:
    for epoch in range(20):
        
        print(min(len(eng_lines_train),subsample)," samples to train")

        
        totloss=0
        model.train()
        for i in range(0,min(subsample,len(eng_lines_train)),b):
            #src = torch.LongTensor(eng_lines_train[i]).unsqueeze(1).to(device)
            #trg = torch.LongTensor(hin_lines_train[i]).unsqueeze(1).to(device)
            src = torch.LongTensor(eng_lines_train[i:i+b]).to(device).permute(1,0)
            trg = torch.LongTensor(hin_lines_train[i:i+b]).to(device).permute(1,0)

            op = model(src,trg[:-1,:])
            #print(op.shape)
            op = op.reshape(-1,vocab_len)
            #print(op.shape)

            trg = trg[1:].reshape(-1)
            #print(op,trg)

            optimizer.zero_grad()
            loss = criterion(op, trg)
            totloss+=loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if i%100==0:
                print(i)
                model.eval()
                xx = translate_sentence(model, eng_lines_train[1])
                print((' '.join(xx)).encode("utf-8"))
                f.write((' '.join(xx)+'\n').encode("utf-8"))
        print("epoch ",epoch,"| loss :",totloss)
    torch.save(model, "./Model/transformer1.pth")
else:
    model = torch.load("./Model/transformer1.pth")

f.close()


print("testing")

import nltk

K=3

f=open("trans_test.txt",'wb')
model.eval()
tt=0
for i in range(0,100):
    eng_sentence = ' '.join([tokenizer_english.index_word[idx] for idx in eng_lines_train[i]])
    xx = translate_sentence(model, eng_lines_train[i])
    #print((' '.join(xx)).encode("utf-8"))
    f.write((eng_sentence+'\n').encode("utf-8"))
    f.write((' '.join(xx)+'\n').encode("utf-8"))
    print(i)
    actual = []
    for j in hin_lines_train[i]:
        if j!=0:
            actual.append(tokenizer_hindi.index_word[j])
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([actual], xx, weights=[1.0/K]*K)
    tt+=BLEUscore
    #print(BLEUscore)
print(tt/100)
f.close()

f=open("trans_test.txt",'wb')
tt=0
model.eval()
for i in range(0,100):
    eng_sentence = ' '.join([tokenizer_english.index_word[idx] for idx in eng_lines_dev[i]])
    xx = translate_sentence(model, eng_lines_dev[i])
    #print((' '.join(xx)).encode("utf-8"))
    f.write((eng_sentence+'\n').encode("utf-8"))
    f.write((' '.join(xx)+'\n').encode("utf-8"))
    print(i)
    actual = []
    for j in hin_lines_dev[i]:
        if j!=0:
            actual.append(tokenizer_hindi.index_word[j])
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([actual], xx, weights=[1.0/K]*K)
    tt+=BLEUscore
    #print(BLEUscore)
print(tt/100)
f.close()
