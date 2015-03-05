
import numpy as np
#from scipy.sparse
from sklearn.datasets import load_iris
import pyvw
import random

class perceptron():
    def __init__(self,data):
        self.b=2**18
        self.k=2+np.max(data.target)
        n,d=data.data.shape
        self.w=np.zeros((self.b,self.k))

    def learn(self, example):
        k=self.k
        w=self.w
        preds=np.zeros(k)
        for f,v in ex.iter_features():
            for j in xrange(k):
                preds[j] += w[f%self.b,j]*v
        yhat = int(np.argmax(preds))
        y = example.get_multiclass_label()-1
        if yhat != y:
            for f,v in ex.iter_features():
                w[f%self.b,yhat] -= v
                w[f%self.b,y]    += v
        return yhat+1

# initialize VW as usual, but use 'pythonbaselearner' as base learner
vw = pyvw.vw("--pythonbaselearner")

data=load_iris()
# tell VW to construct your base learner
p = perceptron(data)
baseLearner = vw.set_base_learner(p)

n,d=data.data.shape
a=range(n)
for t in xrange(10):
    random.shuffle(a)
    for i in a:
        buf= ("%u |0 "%(1+data.target[i])) +' '.join(["%d:%g"%(j,data.data[i,j]) for j in xrange(d)])
        ex = vw.example(buf,labelType=pyvw.vw.lMulticlass)
        ex.learn()
        print "%u %u"%(ex.get_multiclass_label(),ex.get_multiclass_prediction())

#print "%g"%(ex.get_simplelabel_prediction())
