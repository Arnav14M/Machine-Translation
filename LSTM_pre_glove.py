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

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def cleanup(lines_raw):
    lines=[]
    for i in lines_raw:
        s='start '
        for x in i:
            if (x not in string.punctuation) and x!='\n':
                s+=x
            else:
                s+=' '
        lines.append(s)
    return lines

def cleanup2(lines_raw):
    lines=[]
    for i in lines_raw:
        s='शुरू '
        for x in i:
            if (x not in string.punctuation) and x!='\n':
                s+=x
            else:
                s+=' '
        lines.append(s)
    return lines

'''def make_index(lines,d):
    data=[]
    for i in lines:
        l=[]
        for w in i.split():
            l.append(d[w])
        data.append(l)
    return data

def make_one_hot(i,ll):
    v = np.array(i)
    a = np.zeros((len(i),ll))
    a[np.arange(len(i)),v]=1
    return a'''

def make_eng_vector(line):
    vec = []
    suc=1
    for word in line[0].split():
        if word.lower() in eng_word_vec:
            vec.append(eng_word_vec[word.lower()])
        else:
            suc=0
            #vec.append(eng_word_vec[eng_ind_word[1234]])
            #print(word.encode())
    return suc,np.array(vec)

def make_hin_vector(line):
    vec = []
    suc=1
    for word in line[0].split():
        if word in hin_word_vec:
            vec.append(hin_word_vec[word])
        else:
            suc=0
            #vec.append(hin_word_vec[hin_ind_word['1234']])
            #print(word.encode())
    return suc,np.array(vec)
    

device = 'cuda'

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
hin_lines=cleanup2(hin_lines_raw)

eng_lines_dev=cleanup(eng_lines_raw_dev)
hin_lines_dev=cleanup2(hin_lines_raw_dev)

print("Data loaded and cleaned.")

f = open('english_dictionary','rb')
eng_word_vec,eng_ind_word,eng_allvecs = p.load(f)
#eng_word_vec['<start>'] = np.zeros(300)
#eng_word_vec['<end>'] = np.ones(300)
f.close()

f = open('hindi_dictionary','rb')
hin_word_vec,hin_ind_word,hin_allvecs = p.load(f)
#hin_word_vec['<start>'] = np.zeros(300)
#hin_word_vec['<end>'] = np.ones(300)
f.close()

print("GloVe embeddings loaded.")

traindata = (eng_lines[:3000])
trainlabels = (hin_lines[:3000])

devdata = (eng_lines_dev[:100])
devlabels = (hin_lines_dev[:100])

#print(make_eng_vector(traindata[0]),make_hin_vector(trainlabels[0]))
L1=L2=300

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.enc_lstm = nn.LSTMCell(L1,L2)
        self.H=L2
        

    def forward(self, x):
        h = torch.rand((1,self.H)).to(device)
        c = torch.rand((1,self.H)).to(device)
        for i in range(x.shape[0]):
            h,c = self.enc_lstm(x[i].unsqueeze(0),(h,c))
        return (h,c)
        

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.enc_lstm = nn.LSTMCell(L2,L2)
        self.H=L2
        self.out = nn.Linear(self.H,L2)
        

    def forward(self, h0, c0, oplen):
        op=[]
        h = h0
        c = c0
        start = np.array([hin_word_vec['शुरू']])
        out = torch.from_numpy(start).float().to(device)
        for i in range(oplen):
            h,c = self.enc_lstm(out,(h,c))
            out = self.out(h)
            op.append(out.squeeze())
        return op

#print(x,y)
#print(x.argmax(axis=1),y.argmax(axis=1))

enc = encoder().cuda()
dec = decoder().cuda()

losstype = nn.MSELoss()
optimizer1 = optim.Adam(enc.parameters(), lr=0.01)
optimizer2 = optim.Adam(dec.parameters(), lr=0.01)

batch_size=1
bestep=0
actual=0
print(len(traindata))
TRAIN = 0

if TRAIN:
    for epoch in range(10):
        td,tl = traindata,trainlabels
        trainloss=0
        for i in range(0,len(traindata),batch_size):
            s1, ip = make_eng_vector(td[i:i+batch_size])
            s2, ll = make_hin_vector(tl[i:i+batch_size])


            if s1*s2==1:
                actual+=1
                inputs = torch.from_numpy(ip).float().to(device)
                lab = torch.from_numpy(ll).float().to(device)
                h,c = enc(inputs)
                output = dec(h,c,len(lab))
                
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                loss=0
                for k in range(min(len(output),len(lab))):
                    loss += losstype(output[k], lab[k])
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                trainloss+=loss
                if (i%50==0):
                    print(i,actual)
        print("epoch",epoch,"| loss:",trainloss)

    torch.save(enc, "./Model/LSTM_enc1.pth")
    torch.save(dec, "./Model/LSTM_dec1.pth")

else:
    enc = torch.load("./Model/LSTM_enc1.pth")
    dec = torch.load("./Model/LSTM_dec1.pth")

print("TESTING")



f=open("dump.txt",'wb')

#TEST
for testnum in range(20):
    #testnum=3
    s1, ip = make_eng_vector(traindata[testnum:testnum+batch_size])

    if s1:
    
        inputs = torch.from_numpy(ip).float().to(device)

        h,c = enc(inputs)

        '''
        tt=5
        s2=0
        while s2==0:
            s2, ip2 = make_eng_vector(traindata[tt:tt+batch_size])
            inputs2 = torch.from_numpy(ip2).float().to(device)
            tt+=1


        h2,c2 = enc(inputs2)

        print(h-h2)
        print(c-c2)
        '''

        op = dec(h,c,len(ip)+5)
        print(testnum)
        print(traindata[testnum])
        s=''
        for i in op:
            x=i.detach().cpu().numpy()
            index = np.argmin(np.linalg.norm(hin_allvecs - x,axis=1))
            index = str(index)
            if hin_ind_word[index]=='<END>':
                break
            f.write((hin_ind_word[index]+' ').encode("utf-8"))
            
            s+=hin_ind_word[index]+' '
        f.write(('\n').encode("utf-8"))
        print(s.encode())
        
f.close()
