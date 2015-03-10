import pyvw
from sys import argv
import numpy
import theano
import theano.tensor as T
import random


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, n_in, n_out, learning_rate):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.d=n_in
        self.k=n_out

        print '... building the model'
        # generate symbolic variables for input (x and y represent a
        # minibatch)
        x = T.matrix('x')  # data, presented as rasterized images
        y = T.ivector('y')  # labels, presented as 1D vector of [int] labels


        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # construct the logistic regression class
        # Each MNIST image has size 28*28
        #classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        self.cost = self.negative_log_likelihood(y)

        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=self.cost, wrt=self.W)
        g_b = T.grad(cost=self.cost, wrt=self.b)

        # start-snippet-3
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(self.W, self.W - learning_rate * g_W),
                   (self.b, self.b - learning_rate * g_b)]

        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        print '... compiling to C++'
        self.train_model = theano.function(
            inputs=[x,y],
            outputs=(self.cost,self.y_pred),
            updates=updates,
        )

        self.predict = theano.function(
            inputs=[x],
            outputs=self.y_pred,
        )


        print 'done'

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def learn(self, example):
        x=example.get_extra_data()
        yy=example.get_multiclass_label()
        if 1 <= yy <= self.k:
            y=numpy.array([yy-1],dtype=numpy.int32)
            nll,pred = self.train_model(x,y)
        elif yy == 0xffffffff:
            pred = self.predict(x)
        else:
            raise Exception("example label is not in [1,%d]"%self.k)
        return 1+int(pred)

assert(len(argv)==3)
print "... reading data"
dense_shape=None
words=[]
characters=[]
with open(argv[1],'r') as f:
  for line in f:
    if line.strip() == '':
      words.append(characters)
      characters=[]
    else:
      [label,pixels]=line.split("|")
      pix=numpy.array(map(lambda t:float(t.split(':')[1]), pixels[2:].split()))
      if dense_shape is None:
          dense_shape=pix.shape
      else:
          assert(dense_shape==pix.shape)
      pix=numpy.reshape(pix,(1,pix.shape[0]))
      characters.append((line,pix))
if len(characters) > 0:
  words.append(characters)

print dense_shape
vw=pyvw.vw("--pythonbaselearner")
lr=LogisticRegression(dense_shape[0],26,0.01)
vw.set_base_learner(lr)

random.seed(90210)
verbose=False
for t in xrange(10):
    random.shuffle(words)
    print 'pass %d'%(t+1)
    for word in words:
        correct=[]
        pred=[]
        for c in word:
            line,pix=c
            lab = int(line.split("|")[0])
            ex = vw.example("%d |"%lab,labelType=pyvw.vw.lMulticlass)
            ex.set_extra_data(pix)
            ex.learn()
            correct.append(chr(ord('a')-1+lab))
            pred.append(chr(ord('a')-1+ex.get_multiclass_prediction()))
        if verbose:
            print ' '.join(correct)
            print ' '.join(pred)
            print ''



words=[]
characters=[]
with open(argv[2],'r') as f:
  for line in f:
    if line.strip() == '':
      words.append(characters)
      characters=[]
    else:
      [label,pixels]=line.split("|")
      pix=numpy.array(map(lambda t:float(t.split(':')[1]), pixels[2:].split()))
      if dense_shape is None:
          dense_shape=pix.shape
      else:
          assert(dense_shape==pix.shape)
      pix=numpy.reshape(pix,(1,pix.shape[0]))
      characters.append((line,pix))
if len(characters) > 0:
  words.append(characters)

verbose=True
errors=0
cnt=0
for word in words:
    correct=[]
    pred=[]
    for c in word:
        line,pix=c
        lab = int(line.split("|")[0])
        ex = vw.example(" |",labelType=pyvw.vw.lMulticlass)
        ex.set_extra_data(pix)
        ex.learn()
        pp = ex.get_multiclass_prediction()
        if pp != lab:
            errors+=1
        cnt+=1
        correct.append(chr(ord('a')-1+lab))
        pred.append(chr(ord('a')-1+pp))
    if verbose:
        print ' '.join(correct)
        print ' '.join(pred)
        print ''

print 'test errors = %u / %u (%g)'%(errors,cnt,float(errors)/cnt)
