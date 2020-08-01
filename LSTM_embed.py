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
tokenizer_english = tf.keras.preprocessing.text.Tokenizer(num_words=5000,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer_english.fit_on_texts(eng_lines)

tokenizer_english.word_index['<pad>'] = 0
tokenizer_english.index_word[0] = '<pad>'

eng_lines = tokenizer_english.texts_to_sequences(eng_lines)
eng_lines = tf.keras.preprocessing.sequence.pad_sequences(eng_lines, padding='post')



#hindi
tokenizer_hindi = tf.keras.preprocessing.text.Tokenizer(num_words=5000,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
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

class Translator(nn.Module):
    def __init__(self):
        super(Translator, self).__init__()
        self.drop = .5
        self.H = 500
        self.embed_size = 300
        self.embed_enc = nn.Embedding(vocab_len,self.embed_size)
        self.lstm_enc = nn.LSTM(self.embed_size,self.H,num_layers=1,dropout=self.drop)
        self.dropout = nn.Dropout(.1)

        self.embed_dec = nn.Embedding(vocab_len,self.embed_size)
        self.lstm_dec = nn.LSTM(self.embed_size,self.H,num_layers=1,dropout=self.drop)
        self.out = nn.Linear(self.H,vocab_len)

    def forward(self,src,tgt):
        B = tgt.shape[1]
        
        embedded_ip = self.dropout(self.embed_enc(src))

        #h = torch.randn(1,B,self.H).to(device)
        #c = torch.randn(1,B,self.H).to(device)
        outputs , (hidden,cell) = self.lstm_enc(embedded_ip)

        #encoder over

        finalops = torch.zeros(tgt.shape[0],B,vocab_len).to(device)

        x = tgt[0]

        for i in range(1,tgt.shape[0]):
            x = x.unsqueeze(0)
            embedded_op = self.dropout(self.embed_enc(x))
            outputs , (hidden,cell) = self.lstm_dec(embedded_op,(hidden,cell))
            pred = self.out(outputs).squeeze(0)
            finalops[i] = pred
            next_word = pred.argmax(1)
            if random.random()>1:
                x = next_word
            else:
                x = tgt[i]
        return finalops
        

def translate_s(model, sentence, max_length=50):

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



model = Translator().to(device)
optimizer = optim.Adam(model.parameters(), lr=.001)

criterion = nn.CrossEntropyLoss(ignore_index=0)


print("training")

TRAIN=0
b=64 #batch size

subsample = 30000

f=open("tans_dump.txt",'wb')
if TRAIN:
    for epoch in range(5):
        
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
                xx = translate_s(model, eng_lines_train[1])
                print((' '.join(xx)).encode("utf-8"))
                f.write((' '.join(xx)+'\n').encode("utf-8"))
                model.train()
        print("epoch ",epoch,"| loss :",totloss)
    torch.save(model, "./Model/new_lstm1.pth")
else:
    model = torch.load("./Model/new_lstm1.pth")

f.close()


print("testing")

import nltk

'''f=open("trans_test.txt",'wb')
model.eval()
tt=[0]*4
for i in range(0,100):
    eng_sentence = ' '.join([tokenizer_english.index_word[idx] for idx in eng_lines_train[i]])
    xx = translate_s(model, eng_lines_train[i])
    #print((' '.join(xx)).encode("utf-8"))
    f.write((eng_sentence+'\n').encode("utf-8"))
    f.write((' '.join(xx)+'\n').encode("utf-8"))
    print(i)
    actual = []
    for j in hin_lines_train[i]:
        if j!=0:
            actual.append(tokenizer_hindi.index_word[j])
    for K in range(1,5):
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([actual], xx, weights=[1.0/K]*K)
        tt[K-1]+=BLEUscore
        print(BLEUscore)
print(tt)
f.close()'''

f=open("trans_test.txt",'wb')
tt=[0]*4
model.eval()
for i in range(0,100):
    eng_sentence = ' '.join([tokenizer_english.index_word[idx] for idx in eng_lines_dev[i]])
    xx = translate_s(model, eng_lines_dev[i])
    #print((' '.join(xx)).encode("utf-8"))
    f.write((eng_sentence+'\n').encode("utf-8"))
    f.write((' '.join(xx)+'\n').encode("utf-8"))
    print(i)
    actual = []
    for j in hin_lines_dev[i]:
        if j!=0:
            actual.append(tokenizer_hindi.index_word[j])
    for K in range(1,5):
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([actual], xx, weights=[1.0/K]*K)
        tt[K-1]+=BLEUscore
    #print(BLEUscore)
print(tt)
f.close()
