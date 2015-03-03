
/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD
license as described in the file LICENSE.
 */
#include "learner.h"
#include "global_data.h"

LEARNER::base_learner* pythonbaselearner_setup(vw& all);
struct pythonbaselearner {
    vw* all;
    void* impl;
    void (*predict)(pythonbaselearner&,example&);
    void (*learn)  (pythonbaselearner&,example&);
};
