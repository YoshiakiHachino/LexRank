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
# LexRankで文章を要約する．
#************************* 
def lex_rank(sentences, n, t):
    """
    LexRankで文章を要約する．
    @param  sentences: list
        文章([[w1,w2,w3],[w1,w3,w4,w5],..]のような文リスト)
    @param  n: int
        文章に含まれる文の数
    @param  t: float
        コサイン類似度の閾値(default 0.1)
    @return : list
        LexRank
    """
    cosine_matrix = numpy.zeros((n, n))
    degrees = numpy.zeros((n,))
    l = []

     # 1. 隣接行列の作成
    for i in range(n):
        for j in range(n):
            cosine_matrix[i][j] = idf_modified_cosine(sentences, sentences[i], sentences[j])
	    #print 'matrix',i,j,'=',cosine_matrix[i][j]
	    if cosine_matrix[i][j] > t:
                cosine_matrix[i][j] = 1
                degrees[i] += 1
            else:
                cosine_matrix[i][j] = 0

    # 2.LexRank計算
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
#2文間のコサイン類似度を計算
#************************* 
def idf_modified_cosine(sentences, sentence1, sentence2):
    """
    2文間のコサイン類似度を計算
    @param  sentence1: list
        文1([w1,w2,w3]のような単語リスト)
    @param  sentence2: list
        文2([w1,w2,w3]のような単語リスト)
    @param  sentences: list
        文章([[w1,w2,w3],[w1,w3,w4,w5],..]のような単語リスト)
    @return : float
        コサイン類似度
    """
    tf1 = compute_tf(sentence1)
    tf2 = compute_tf(sentence2)
    idf_metrics = compute_idf(sentences)
    return cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics)
#*************************
#TFを計・
#************************* 
def compute_tf(sentence):
    """
    TFを計算
    @param  sentence: list
        文([w1,w2,w3]のような単語リスト)
    @return : list
        TFリスト
    """
    tf_values = Counter(sentence)
  
    tf_metrics = {}

    max_tf = find_tf_max(tf_values)

    for term, tf in tf_values.items():
        tf_metrics[term] = tf / max_tf

    return tf_metrics
#*************************
#単語の最大出現頻度を探索
#************************* 
def find_tf_max(terms):
    """
    単語の最大出現頻度を探索
    @param  terms: list
        単語の出現頻度
    @return : int
        単語の最大出現頻度
    """
    return max(terms.values()) if terms else 1
#*************************
#文章中の単語のIDF値を計算
#************************* 
def compute_idf(sentences):
    """
    文章中の単語のIDF値を計算
    @param sentences: list
        文章([[w1,w2,w3],[w1,w3,w4,w5],..]のような単語リスト)
    @return: list
        IDFリスト
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
# コサイン類似度を計算
#************************* 
def cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics):
    """
    コサイン類似度を計算
    @param  sentence1: list
        文1([w1,w2,w3]のような単語リスト)
    @param  sentence2: list
        文2([w1,w2,w3]のような単語リスト)
    @param  tf1: list
        文1のTFリスト
    @param  tf2: list
        文2のTFリスト
    @param  idf_metrics: list
        文章のIDFリスト
    @return : float
        コサイン類似度
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
#   べき乗法を行なう
#************************
def power_method(cosine_matrix, n, e):
    """
    べき乗法を行なう
    @param  scosine_matrix: list
        確率行列
    @param  n: int
        文章中の文の数
    @param  e: float
        許容誤差ε
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
			wordlist.append(word[:-1])  #word[-1]は最後の文字 word[:-1]は最後の文字を除く
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