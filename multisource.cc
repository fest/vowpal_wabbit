#include "multisource.h"
#include "simple_label.h"
#include "timing.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <stdio.h>

int really_read(int sock, void* in, size_t count)
{
  char* buf = (char*)in;
  size_t done = 0;
  int r= 0;
  while (done < count)
    {
      struct timeval tv=tic();
      if ((r = read(sock,buf,count-done)) == 0){
	toc(&tv,&timers[REALLY_READ]);
	return 0;
      }
      else{
	toc(&tv,&timers[REALLY_READ]);
	if (r < 0)
	  {
	    cerr << "argh! bad read! " << endl;
	    perror(NULL);
	    exit(0);
	  }
	else
	  {
	    done += r;
	    buf += r;
	  }
      }
    }
  return done;
}

bool blocking_get_prediction(int sock, prediction &p)
{
  int count = really_read(sock, &p, sizeof(p));
  bool ret = (count == sizeof(p));
  return ret;
}

bool blocking_get_global_prediction(int sock, global_prediction &p)
{
  int count = really_read(sock, &p, sizeof(p));
  bool ret = (count == sizeof(p));
  return ret;
}

void send_prediction(int sock, prediction &p)
{
  struct timeval tv=tic();
  if (write(sock, &p, sizeof(p)) < (int)sizeof(p))
    {
      cerr << "argh! bad write! " << endl;
      perror(NULL);
      exit(0);
    }
  toc(&tv,&timers[SEND_PREDICTION]);
}

void send_global_prediction(int sock, global_prediction p)
{
  struct timeval tv=tic();
  if (write(sock, &p, sizeof(p)) < (int)sizeof(p))
    {
      cerr << "argh! bad write! " << endl;
      perror(NULL);
      exit(0);
    }
  toc(&tv,&timers[SEND_GLOBAL_PREDICTION]);
}

void reset(partial_example &ex)
{
  ex.features.erase();
}

int receive_features(parser* p, void* ex)
{
  example* ae = (example*)ex;
  io_buf* input = p->input;
  fd_set fds;
  FD_ZERO(&fds);
  for (int* sock= input->files.begin; sock != input->files.end; sock++)
    FD_SET(*sock,&fds);
  
  while (input->files.index() > 0)
    {
      struct timeval tv=tic();
      if (select(p->max_fd,&fds,NULL, NULL, NULL) == -1)
	{
	  cerr << "Select failed!" << endl;
	  perror(NULL);
	  exit (1);
	}
      toc(&tv,&timers[SELECT]);
      for (int index = 0; index < (int)input->files.index(); index++)
	{
	  int sock = input->files[index];
	  if (FD_ISSET(sock, &fds))
	    {//there is a feature or label to read
	      prediction pre;
	      if (!blocking_get_prediction(sock, pre) )
		{
		  FD_CLR(sock, &fds);
		  close(sock);
		  memmove(input->files.begin+index,
			  input->files.begin+index+1,
			  (input->files.index() - index-1)*sizeof(int));
		  input->files.pop();
		  memmove(p->ids.begin+index, 
			  p->ids.begin+index+1, 
			  (p->ids.index() - index-1)*sizeof(size_t));
		  p->ids.pop();
		  memmove(p->counts.begin+index,
			  p->counts.begin+index+1,
			  (p->counts.index() - index-1)*sizeof(size_t));
		  p->counts.pop();
		  index--;
		}
	      else
		{
		  if (pre.example_number != ++ (p->counts[index]))
		    cout << "count is off! " << pre.example_number << " != " << p->counts[index] << 
		      " for source " << index << " prediction = " << pre.p << endl;
		  if (pre.example_number == p->finished_count + ring_size - 1)
		    FD_CLR(sock,&fds);//this ones to far ahead, let the buffer fill for awhile.
		  size_t ring_index = pre.example_number % p->pes.index();
		  if (p->pes[ring_index].features.index() == 0)
		    p->pes[ring_index].example_number = pre.example_number;
		  if (p->pes[ring_index].example_number != (int)pre.example_number)
		    cerr << "Error, example " << p->pes[ring_index].example_number << " != " << pre.example_number << endl;
		  feature f = {pre.p, p->ids[index]};
		  push(p->pes[ring_index].features, f);
		  if (sock == p->label_sock) // The label source
		    {
		      label_data ld;
		      size_t len = sizeof(ld.label)+sizeof(ld.weight);
		      char c[len];
		      really_read(sock,c,len);
		      bufread_simple_label(&(p->pes[ring_index].ld), c);
		    }

		  if( p->pes[ring_index].features.index() == input->count )
		    {
		      push( ae->indices, multindex );
		      push_many( ae->atomics[multindex], p->pes[ring_index].features.begin, p->pes[ring_index].features.index() );
		      label_data* ld = (label_data*)ae->ld;
		      *ld = p->pes[ring_index].ld;
		      reset(p->pes[ring_index]);
		      p->finished_count++;
		      return ae->atomics[multindex].index();
		    }
		}
	    }
	  else  if (p->counts[index] < p->finished_count + ring_size -1)
	    FD_SET(sock,&fds);
	}
    }
  return 0;
}

#ifdef USE_EPOLL
int receive_features_epoll(parser* p, void* ex)
{
  example* ae = (example*)ex;
  io_buf* input = p->input;

  while(p->nclosed < input->files.index())
    {
      int nfds = epoll_wait(p->epfd, p->events, input->files.index(), -1);
      if (nfds < 0)
	{
          cerr << "epoll_wait failed!" << endl;
	  perror(NULL);
          exit(1);
	}
      for(int i = 0; i < nfds; i++) 
	{
	  int_pair* ip = (int_pair*)p->events[i].data.ptr;
	  int sock = ip->fd;
	  int index = ip->id;
	  prediction pre;
	  if (!blocking_get_prediction(sock, pre) )
	    {
	      p->nclosed+=1;
	      close(sock); /* Removes sock from epoll */
	    }
	  else
	    {
	      if (pre.example_number != ++ (p->counts[index]))
		cout << "count is off! " << pre.example_number << " != " << p->counts[index] << 
		  " for source " << index << " prediction = " << pre.p << endl;
	      if (pre.example_number == p->finished_count + ring_size - 1)
		{
		  /* this one is too far ahead, let the buffer fill for awhile. */
		  if (epoll_ctl(p->epfd, EPOLL_CTL_DEL, p->input->files[index], NULL)!=0)
		    perror(NULL);
		  push(p->toofar,index);
		}
	      size_t ring_index = pre.example_number % p->pes.index();
	      if (p->pes[ring_index].features.index() == 0)
		p->pes[ring_index].example_number = pre.example_number;
	      if (p->pes[ring_index].example_number != (int)pre.example_number)
		cerr << "Error, example " << p->pes[ring_index].example_number << " != " << pre.example_number << endl;
	      feature f = {pre.p, p->ids[index]};
	      push(p->pes[ring_index].features, f);
	      if (sock == p->label_sock) // The label source
		{
		  label_data ld;
		  size_t len = sizeof(ld.label)+sizeof(ld.weight);
		  char c[len];
		  really_read(sock,c,len);
		  bufread_simple_label(&(p->pes[ring_index].ld), c);
		}
	      if( p->pes[ring_index].features.index() == input->count )
		{
		  push( ae->indices, multindex );
		  push_many( ae->atomics[multindex], p->pes[ring_index].features.begin, p->pes[ring_index].features.index() );
		  label_data* ld = (label_data*)ae->ld;
		  *ld = p->pes[ring_index].ld;
		  reset(p->pes[ring_index]);
		  p->finished_count++;
		  //put everyone back in epoll
		  while(!p->toofar.empty())
		    {
		      int j = p->toofar.pop();
		      struct epoll_event ev;
		      int_pair ip = {p->input->files[j], j};
		      ev.data.ptr = &ip;
		      ev.events = EPOLLIN;
		      if (epoll_ctl(p->epfd, EPOLL_CTL_ADD, p->input->files[j], &ev)!=0)
			perror(NULL);
		    }
		  return ae->atomics[multindex].index();
		}
	    }
	}
    }
  return 0;
}
#endif

