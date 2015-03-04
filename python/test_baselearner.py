import pyvw

class BinaryBaseLearner():
    def __init__(self):
        return

    def learn(self, example):
        print "python is learning!"
        return 2.0
    
# initialize VW as usual, but use 'pythonbaselearner' as base learner
vw = pyvw.vw("--quiet --pythonbaselearner")

# tell VW to construct your base learner
pyb = BinaryBaseLearner()
baseLearner = vw.set_base_learner(pyb)

ex = vw.example("1 |x a b |y c")

# the example goes into VW, back out to BinaryBaseLearner, 
# back to VW, and back out here
ex.learn()

print "%g"%(ex.get_simplelabel_prediction())
#print "%u"%(ex.get_multiclass_prediction())
