#pragma once
#ifndef __FILEH__
#define __FILEH__

//#define WIN32_LEAN_AND_MEAN
//#include <Windows.h>
////#include <stdint.h> // portable: uint64_t   MSVC: __int64 
//
//// MSVC defines this in winsock2.h!?
//typedef struct timeval {
//    long tv_sec;
//    long tv_usec;
//} timeval;
//
//int gettimeofday(struct timeval* tp);

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
    void initVector(unsigned int** vec_h, unsigned int size, unsigned int num_bins);
    void verify(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins);
       void startTime(Timer* timer);
       void stopTime(Timer* timer);
       float elapsedTime(Timer timer);

#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif