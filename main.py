#-*- coding:utf-8 -*-
from classify import Classify
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding("utf8")

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
    del NBmodel_test
