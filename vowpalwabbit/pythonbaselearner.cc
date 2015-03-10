/*
   Copyright (c) by respective owners including Yahoo!, Microsoft, and
   individual contributors. All rights reserved.  Released under a BSD (revised)
   license as described in the file LICENSE.
   */
#include "pythonbaselearner.h"

using namespace std;
using namespace LEARNER;

namespace
{
  pythonbaselearner* myb;

void 
learn(pythonbaselearner& b, base_learner& unused, example& ec) {
    if(b.learn)
        b.learn(b,ec);
    else if (!b.all->quiet)
        cerr<< "learn not initialized" << endl;
}

void 
save_load(pythonbaselearner& b, io_buf& model_file, bool read, bool text) 
{
    if (!read && !b.all->quiet)
        cerr << "python base learner does not save (or load) a vw regressor. Use python for state"<< endl;
}

void 
finish_example(vw& all, pythonbaselearner& b, example& ec) {
    if (ec.python.decref && ec.python.extra) {
      ec.python.decref(ec.python.extra);
      ec.python.copy = 0;
      ec.python.decref = 0;
      ec.python.extra = 0;
    }
}

}

pythonbaselearner*
get_pythonbaselearner ()
{
  return myb;
}

base_learner* pythonbaselearner_setup(vw& all) 
{
  if (missing_option(all, false, "pythonbaselearner", "python base learner")) 
    return NULL;
  
  myb = calloc_or_die<pythonbaselearner> (1);
  myb->all = &all;

  if (!all.quiet) {
    cerr << "Enabling pythonbaselearner" << endl;
  }
  
  learner<pythonbaselearner>& l = init_learner(myb, learn, 1);
  l.set_save_load(save_load);
  l.set_finish_example(finish_example);
  return make_base(l);
}
