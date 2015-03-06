import cPickle
import gzip
from math import sqrt
import numpy 
import os
import pyvw
import random
import sys
import theano
import theano.tensor as T

def load_data():
    #############
    # LOAD DATA #
    #############

    dataset='mnist.pkl.gz'

    # Download the MNIST dataset if it is not present
    if (not os.path.isfile(dataset)):
        import urllib

        def dlProgress(count, blockSize, totalSize):
          percent = int(count*blockSize*100/totalSize)
          sys.stdout.write("\r" + "...%d%%" % percent)
          sys.stdout.flush()

        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset,reporthook=dlProgress)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

class LogisticRegression(object):
    def __init__(self, n_in, n_out):
        print '... building model'
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        x = T.vector('x')
        y = T.lvector('y')
        eta = T.scalar('eta')
        p_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        cost = -T.mean(T.log(p_y_given_x)[theano.shared(numpy.array([0])), y])
        g_W = T.grad(cost=cost, wrt=self.W)
        g_b = T.grad(cost=cost, wrt=self.b)
        updates = [(self.W, self.W - eta * g_W),
                   (self.b, self.b - eta * g_b)]
        self.train = theano.function(inputs=[x,y,eta], 
                                     outputs=T.argmax(p_y_given_x),
                                     updates=updates)

    def learn(self, example):
        x,y,eta=example.get_extra_data()
        ypred=self.train(x,y,eta)
        return int(ypred+1)

if __name__ == '__main__':
    train,val,test=load_data()
    vw=pyvw.vw("--quiet --pythonbaselearner")
    baseLearner=LogisticRegression(28*28,10)
    vw.set_base_learner(baseLearner)
    ex = vw.example(" | ",labelType=pyvw.vw.lMulticlass)

    trainx,trainy = train
    flassy=numpy.reshape(0,(1,))
    n,d=trainx.shape
    a=range(n)
    cnt=0
    loss=0.0
    sincelastcnt=0
    sincelastloss=0.0
    nextprint=1
    eta=0.1
    initial_t=5000

    print "%7.7s %7.7s %7.7s"%("example","average","  since")
    print "%7.7s %7.7s %7.7s"%("counter","   loss","   last")

    for t in xrange(10):
      random.shuffle(a)
      for i in a:
        flassy[0]=trainy[i]
        datum=(trainx[i],flassy,eta*sqrt(initial_t)/sqrt(initial_t+cnt))
        ex.set_extra_data(datum)
        ex.learn()

        if (1+trainy[i] != ex.get_multiclass_prediction()):
          loss=loss+1
          sincelastloss=sincelastloss+1
        cnt=cnt+1
        sincelastcnt=sincelastcnt+1
        if (cnt > nextprint):
          print "%7u %7.5f %7.5f"%(cnt, loss/cnt, sincelastloss/sincelastcnt)
          nextprint=2*nextprint
          sincelastcnt=0
          sincelastloss=0.0
