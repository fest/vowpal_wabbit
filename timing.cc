#include "timing.h"

struct timeval timers[6]={0};

long double
tvtoldbl (const struct timeval* x)
{
  return x->tv_sec + 1.0E-6 * x->tv_usec;
}

void add_diff (struct timeval* start, struct timeval* finish, struct timeval *acc)
{
  if (finish->tv_usec < start->tv_usec) {
      finish->tv_usec += 1000000;
      finish->tv_sec -= 1;
  }

  acc->tv_usec += finish->tv_usec - start->tv_usec;
  acc->tv_sec  += finish->tv_sec  - start->tv_sec;
  if (acc->tv_usec >= 1000000){
      acc->tv_usec -= 1000000;
      acc->tv_sec  += 1;
  }
}

struct timeval tic(){
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv;
}

void toc(struct timeval *t, struct timeval *acc){
    struct timeval tv;
    gettimeofday(&tv, 0);
    add_diff(t,&tv,acc);
}


