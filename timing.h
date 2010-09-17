#ifndef TIMING_H
#define TIMING_H

#define REALLY_READ 0
#define SEND_PREDICTION 1
#define SEND_GLOBAL_PREDICTION 2
#define SELECT 3
#define IO_BUF_READ 4
#define IO_BUF_WRITE 5



#include <sys/time.h>

long double tvtoldbl(const struct timeval* x);
struct timeval tic();
void toc(struct timeval *t, struct timeval *acc);

extern struct timeval timers[];
#endif // TIMING_H
