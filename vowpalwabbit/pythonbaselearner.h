
/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD
license as described in the file LICENSE.
 */
#include "learner.h"
#include "global_data.h"

struct pythonbaselearner {
  vw*    all;
  void*  impl;
  void (*learn)  (pythonbaselearner&,example&);
};

LEARNER::base_learner* pythonbaselearner_setup(vw& all);
pythonbaselearner* get_pythonbaselearner ();
