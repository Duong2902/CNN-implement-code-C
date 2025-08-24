#ifndef _KERNEL_H_
#define _KERNEL_H_
#include "stdio.h"
#include "stdlib.h"
#include "time.h"

typedef struct
{
    int size;
    int channels;
    int filters;
    float ****kernel_value;
} KERNEL; 

KERNEL init_kernel (int size, int channels, int filters);

void initialize_kernel_values(KERNEL *kernel);

void print_kernel_values(const KERNEL *kernel);

void free_kernel (KERNEL* kernel);

void load_kernel_from_file(KERNEL* kernel, const char* filename, int size, int channels, int filters);

void copy_kernel(KERNEL *dest, KERNEL *src) ;

void copy_bias(float **dest, float *src, int size) ;

#endif