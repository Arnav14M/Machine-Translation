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

'''
f = open('data/hin_vec/hi.tsv','r',encoding='utf-8')

x = f.readlines()
f.close()

hin_allvecs = []

hin_word_vec = {}
hin_ind_word = {}
i=0
while i<len(x):
    index,word,vec = x[i].split('\t')
    width = len(vec[1:].split())
    i=i+1
    for j in range(int(300/width)-1):
        vec+=x[i]
        i=i+1
    vec = np.array(vec[1:-2].split()).astype('float64')
    hin_word_vec[word] = vec
    hin_ind_word[index] = word
    hin_allvecs.append(vec)

hin_allvecs = np.array(hin_allvecs)

with open('hindi_dictionary', 'wb') as yeet:
    p.dump((hin_word_vec,hin_ind_word,hin_allvecs), yeet)
'''

f = open('data/eng_vec/glove.6B.300d.txt','r',encoding="utf8")

eng_allvecs = []

eng_word_vec = {}
eng_ind_word = {}
i=0
gog = f.readlines()
print(len(gog))
for x in gog:
    sp = x.split()
    word = str(sp[0])
    vec = np.array(sp[1:]).astype('float64')
    eng_word_vec[word] = vec.copy()
    eng_ind_word[i] = word
    eng_allvecs.append(vec)
    i+=1
    if (i%100)==0:
        print(i)
f.close()

f = open('english_dictionary','wb')
p.dump((eng_word_vec,eng_ind_word,eng_allvecs),f)
f.close()
