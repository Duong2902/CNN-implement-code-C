#ifndef _FEATURE_MAP_H_
#define _FEATURE_MAP_H_

#include "stdio.h"
#include "stdlib.h"
#include "time.h"

typedef struct
{
    int batch_size;
    int height;
    int width;
    int channels;
    float ****fm_value;
} FEATURE_MAP;

FEATURE_MAP init_feature_map(int height, int width, int channels, int batch_size);

void initialize_input_values(FEATURE_MAP *input);

void print_feature_map_values(const FEATURE_MAP *input, char* str); 

void free_feature_map(FEATURE_MAP *input);

float output_pixel_value(float* output_gemm, int height, int width);

FEATURE_MAP padding_feature_map(FEATURE_MAP input, int padding);

FEATURE_MAP* init_feature_map_array(int num, int height, int width, int channels);

void free_feature_map_array(FEATURE_MAP* array, int num);

#endif // _FEATURE_MAP_H_