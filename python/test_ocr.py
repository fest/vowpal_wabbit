import pyvw
from sys import argv;

class SequenceLabeler(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        sch.set_options( sch.AUTO_HAMMING_LOSS | sch.AUTO_CONDITION_FEATURES )

    def _run(self, word):
        output = []
        loss=0
        for n in range(len(word)):
            vwline = word[n]
            with self.vw.example(vwline,labelType=pyvw.vw.lMulticlass) as ex:
                pred = self.sch.predict(examples=ex, my_tag=n+1, oracle=ex.get_multiclass_label(), condition=(n,'a'))
                loss += (pred != ex.get_multiclass_label())
                output.append(pred)

        return (loss, output)

print "... reading data"
words=[]
characters=[]
with open(argv[1],'r') as f:
  for line in f:
    if line.strip() == '':
      words.append(characters)
      characters=[]
    else:
      characters.append(line)

    if len(words) > 1000:
      break
if len(characters) > 0:
  words.append(characters)

vw=pyvw.vw("--ring_size 1000 --search 26 --search_task hook --search_alpha 5e-5 -l 0.25 -b 24 -q pp --nn 1 --inpass --search_history_length 2")

sequenceLabeler = vw.init_search_task(SequenceLabeler)
sequenceLabeler.learn(words[1:])
(loss, output) = sequenceLabeler.predict(words[0])
print "%u %u"%(loss,len(words[0]))
