#include "wtime.h"
#ifndef WIN32
#include <sys/time.h>
#else
#include <time.h>
#include <Windows.h>
#endif
#include <stdlib.h>

/*  Prototype  */
void wtime(float *t)
{
  static int sec = -1;

  struct timeval tv;
  gettimeofday(&tv, 0);
  if (sec < 0)
        sec = tv.tv_sec;
  *t = (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}

/*****************************************************************/
/******         E  L  A  P  S  E  D  _  T  I  M  E          ******/
/*****************************************************************/
float elapsed_time( void )
{
    float t;

    wtime( &t );
    return( t );
}

float start[64], elapsed[64];

/*****************************************************************/
/******            T  I  M  E  R  _  C  L  E  A  R          ******/
/*****************************************************************/
void timer_clear( int n )
{
    elapsed[n] = 0.0;
}


/*****************************************************************/
/******            T  I  M  E  R  _  S  T  A  R  T          ******/
/*****************************************************************/
void timer_start( int n )
{
    start[n] = elapsed_time();
}


/*****************************************************************/
/******            T  I  M  E  R  _  S  T  O  P             ******/
/*****************************************************************/
void timer_stop( int n )
{
    float t, now;

    now = elapsed_time();
    t = now - start[n];
    elapsed[n] += t;

}

/*****************************************************************/
/******            T  I  M  E  R  _  R  E  A  D             ******/
/*****************************************************************/
float timer_read( int n )
{
    return( elapsed[n] );
}
