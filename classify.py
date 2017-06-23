#-*- coding:utf-8 -*-
from __future__ import division
import gensim
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import jieba
import jieba.posseg as pseg

import sys
reload(sys)
sys.setdefaultencoding("utf8")

class Classify:
    def __del__(self):
        print("----delete object----") 
    def load_W2V_Model(self,ModelName):
        '''
        load word2vec model
        '''
        self.model = gensim.models.word2vec.Word2Vec.load(ModelName)
        
    def w2v(self,words):
        Vec = []
        num = len(words)
        for word in words:
            vec = self.model[word]
            Vec.append(vec)
            #print(vec)
        Vec = sum(Vec)/num
        print(Vec)
        return Vec

    def sentence2vec(self, sentences):
        Vec = []
        for sentence in sentences:
            words = self.jieba_cut(sentence)
            vec = self.w2v(words)
            Vec.append(vec)

        return Vec

    def train(self,X_train,Y_train):
        Vec = self.sentence2vec(X_train)
        self.clf = GaussianNB()
        self.clf.fit(Vec,Y_train)

    def predict(self,Test):
        Vec = self.sentence2vec(Test)
        result = self.clf.predict(Vec)
        print(result)
        
    def jieba_cut(self,sentence):
        jieba.load_userdict("AllMusicLibrary.txt")
        words_flag = pseg.cut(sentence)
	words = []
        for word, flag in words_flag:
            words.append(word)
        return words


    def save_NBmodel(self, model_name):
        #保存模型
        joblib.dump(self.clf, model_name)
        print("Save Seccessfully")
   
    def load_NBmodel(self,model_name,word2vecModel = "word2vec.model"):
        #加载模型
        self.load_W2V_Model(word2vecModel)
        self.clf = joblib.load(model_name)

'''
if __name__ == "__main__":

    X_train = np.array([u"我想听张学友的歌",u"周杰伦的龙卷风",u"鹿晗有什么歌好听",u"姚明打篮球好厉害",u"张继科会打乒乓球",u"詹姆士是体育明星"])
    Y_train = np.array([1,1,1,2,2,2])
    Test_data = [u"我想听薛之谦的演员","邓亚萍是体育明星","刘翔是体育明星"]
    Model = Classify()
    Model.load_W2V_Model("word2vec.model")
    Model.train(X_train,Y_train)
    Model.predict(Test_data)

    Model.save_NBmodel( "NB.model")
    del Model

    NBmodel_test = Classify()
    NBmodel_test.load_NBmodel("NB.model")
    NBmodel_test.predict(Test_data)
'''
