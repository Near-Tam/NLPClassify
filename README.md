# NLPClassify
基于朴素贝叶斯算法进行文本分类

#安装库
1.sklearn
2.jieba
3.gensim
4.numpy

#导入模块
from classify import Classify
import numpy as np

#训练
#1.导入数据
    X_train = np.array([u"我想听张学友的歌",u"周杰伦的龙卷风",u"鹿晗有什么歌好听",u"姚明>打篮球好厉害",u"张继科会打乒乓球",u"詹姆士是体育明星"])
    Y_train = np.array([1,1,1,2,2,2])
    Test_data = [u"我想听薛之谦的演员","邓亚萍是体育明星","刘翔是体育明星"]

#2.加载word2vec模型
    Model = Classify()
    Model.load_W2V_Model("word2vec.model")

#3.训练模型
    Model.train(X_train,Y_train)

#4.利用模型进行预测
    Model.predict(Test_data)

#5.保存模型
    Model.save_NBmodel( "NB.model")

#6.加载模型并使用
    NBmodel_test = Classify()
    NBmodel_test.load_NBmodel("NB.model")
    NBmodel_test.predict(Test_data)
