#_*_coding:utf-8_*_
import matplotlib.pyplot as plt
import pandas as pd
import numpy 
import sys
import re
import scipy.linalg

import math
from collections import Counter
"""
a = [4,5,2,1,3]
b = ['aaa','bbb','ccc','ddd','eee']

ab=zip(a,b)
print ab
abc=sorted(ab,reverse=True)

print abc
"""
#words=['aaa','bbb','ccc.','ddd','fff','ggg.','hhh','iii','jjj','kkk,','lll','mmm:','nnn','ooo.','ppp','qqq!','sss.','ttt','uuu','vvv.']
#*************************
# LexRank�ŕ��͂�v�񂷂�D
#************************* 
def lex_rank(sentences, n, t):
    """
    LexRank�ŕ��͂�v�񂷂�D
    @param  sentences: list
        ����([[w1,w2,w3],[w1,w3,w4,w5],..]�̂悤�ȕ����X�g)
    @param  n: int
        ���͂Ɋ܂܂�镶�̐�
    @param  t: float
        �R�T�C���ގ��x��臒l(default 0.1)
    @return : list
        LexRank
    """
    cosine_matrix = numpy.zeros((n, n))
    degrees = numpy.zeros((n,))
    l = []

     # 1. �אڍs��̍쐬
    for i in range(n):
        for j in range(n):
            cosine_matrix[i][j] = idf_modified_cosine(sentences, sentences[i], sentences[j])
	    #print 'matrix',i,j,'=',cosine_matrix[i][j]
	    if cosine_matrix[i][j] > t:
                cosine_matrix[i][j] = 1
                degrees[i] += 1
            else:
                cosine_matrix[i][j] = 0

    # 2.LexRank�v�Z
    for i in range(n):
        for j in range(n):
            cosine_matrix[i][j] = cosine_matrix[i][j] / degrees[i]

    #ratings = power_method(cosine_matrix, n,1000 ) #m.nakai add 1000
    ratings,vals,vecs=EigenValue(cosine_matrix)
    #print sentences
    #print ratings
    print ratings
    print vals
    #print ratings.index(max(ratings))
    
    return zip(vals,sentences)
#*************************
#2���Ԃ̃R�T�C���ގ��x���v�Z
#************************* 
def idf_modified_cosine(sentences, sentence1, sentence2):
    """
    2���Ԃ̃R�T�C���ގ��x���v�Z
    @param  sentence1: list
        ��1([w1,w2,w3]�̂悤�ȒP�ꃊ�X�g)
    @param  sentence2: list
        ��2([w1,w2,w3]�̂悤�ȒP�ꃊ�X�g)
    @param  sentences: list
        ����([[w1,w2,w3],[w1,w3,w4,w5],..]�̂悤�ȒP�ꃊ�X�g)
    @return : float
        �R�T�C���ގ��x
    """
    tf1 = compute_tf(sentence1)
    tf2 = compute_tf(sentence2)
    idf_metrics = compute_idf(sentences)
    return cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics)
#*************************
#TF���v�E
#************************* 
def compute_tf(sentence):
    """
    TF���v�Z
    @param  sentence: list
        ��([w1,w2,w3]�̂悤�ȒP�ꃊ�X�g)
    @return : list
        TF���X�g
    """
    tf_values = Counter(sentence)
  
    tf_metrics = {}

    max_tf = find_tf_max(tf_values)

    for term, tf in tf_values.items():
        tf_metrics[term] = tf / max_tf

    return tf_metrics
#*************************
#�P��̍ő�o���p�x��T��
#************************* 
def find_tf_max(terms):
    """
    �P��̍ő�o���p�x��T��
    @param  terms: list
        �P��̏o���p�x
    @return : int
        �P��̍ő�o���p�x
    """
    return max(terms.values()) if terms else 1
#*************************
#���͒��̒P���IDF�l���v�Z
#************************* 
def compute_idf(sentences):
    """
    ���͒��̒P���IDF�l���v�Z
    @param sentences: list
        ����([[w1,w2,w3],[w1,w3,w4,w5],..]�̂悤�ȒP�ꃊ�X�g)
    @return: list
        IDF���X�g
    """
    idf_metrics = {}
    sentences_count = len(sentences)

    for sentence in sentences:
        for term in sentence:
            if term not in idf_metrics:
                n_j = sum(1 for s in sentences if term in s)
                idf_metrics[term] = math.log(sentences_count / (1.0 + n_j))

    return idf_metrics
#*************************
# �R�T�C���ގ��x���v�Z
#************************* 
def cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics):
    """
    �R�T�C���ގ��x���v�Z
    @param  sentence1: list
        ��1([w1,w2,w3]�̂悤�ȒP�ꃊ�X�g)
    @param  sentence2: list
        ��2([w1,w2,w3]�̂悤�ȒP�ꃊ�X�g)
    @param  tf1: list
        ��1��TF���X�g
    @param  tf2: list
        ��2��TF���X�g
    @param  idf_metrics: list
        ���͂�IDF���X�g
    @return : float
        �R�T�C���ގ��x
    """
    unique_words1 = set(sentence1)
    unique_words2 = set(sentence2)
    common_words = unique_words1 & unique_words2

    numerator = sum((tf1[t] * tf2[t] * idf_metrics[t] ** 2) for t in common_words)
    denominator1 = sum((tf1[t] * idf_metrics[t]) ** 2 for t in unique_words1)
    denominator2 = sum((tf2[t] * idf_metrics[t]) ** 2 for t in unique_words2)

    if denominator1 > 0 and denominator2 > 0:
        return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
    else:
        return 0.0  
#*************************
#   �ׂ���@���s�Ȃ�
#************************
def power_method(cosine_matrix, n, e):
    """
    �ׂ���@���s�Ȃ�
    @param  scosine_matrix: list
        �m���s��
    @param  n: int
        ���͒��̕��̐�
    @param  e: float
        ���e�덷��
    @return: list
        LexRank
    """
    transposed_matrix = cosine_matrix.T
    sentences_count = n

    p_vector = numpy.array([1.0 / sentences_count] * sentences_count)

    lambda_val = 1.0

    while lambda_val > e:
        next_p = numpy.dot(transposed_matrix, p_vector)
        lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
        p_vector = next_p

    return p_vector
#*************************
# Eigen Value
#*************************     
def EigenValue(A):
	hi = 2
	lo = 0

	#A = numpy.matrix([[1,2,3],[4,5,6],[7,8,9]])

	#eigen_value,eigen_vector = scipy.linalg.eigh(A,eigvals=(lo,hi))
	eigen_value,eigen_vector = scipy.linalg.eig(A)
	#print 'in EigenValue=>',eigen_value
	eigen_id = numpy.argsort(eigen_value)[::-1]
	#print eigen_id
	#eigen_value = eigen_value[eigen_id]
	#eigen_vector = eigen_vector[:,eigen_id]

	#print eigen_value
	#print eigen_vector
	return (eigen_id,eigen_value,eigen_vector)    
#*************************
# main
#*************************
def main():
	allwords=[]
	f = open('alice.txt','r')
	for line in f:
		words=line[:-1].split()
		for word in words:
			allwords.append(word)
	f.close()
	#print allwords

	wordlist=[]
	sentences=[]
	for word in allwords:
		#print word
		if word.find('.') >=0 or word.find(':') >= 0 or word.find('?') >= 0 or word.find('!') >= 0:
			#print '       ',word
			wordlist.append(word[:-1])  #word[-1]�͍Ō�̕��� word[:-1]�͍Ō�̕���������
			sentences.append(wordlist)
			wordlist =[]
		else:
			wordlist.append(word)

	lc=0
	cc=0
	allower=[]
	for sentence in sentences:
		lc += 1
		if(lc < 0):
			continue;
		allower.append(sentence)
		print sentence
		cc += 1
		if(lc >= 55):
			break;
		#print sentence
	print allower
	
	outs=lex_rank(allower, cc, 0.2)
	outsR = sorted(outs,reverse=True)
	ic = 0
	for item in outsR:
		ic += 1
		rank,bun = item
		sent = ' '.join(bun)
		print sent,'.'
		if(ic > 10):
			break
#*************************
# root
#************************* 	
if __name__ == '__main__':
	s = 'alice.txt'
	args = sys.argv
	if len(args) > 1:
		s = args[1:]
	main()			