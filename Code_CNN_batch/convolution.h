#include "kernel.h"
#include "feature_map.h" 

float *initialize_bias_conv2d(int channels);

float* load_bias(const char* filename, int filters);

FEATURE_MAP Conv2D(FEATURE_MAP input, KERNEL kernel, int padding, int stride, float *bias);

FEATURE_MAP max_pooling(FEATURE_MAP input, int kernel_size, int stride, int**** index_row, int**** index_col);


//FEATURE_MAP* batch_normalize_array(FEATURE_MAP* feature_map_array, int batch_size, float scale, float shift);

float **Flatten(FEATURE_MAP input, int flat_dim);

void free_Flatten(float **input, int size);

float* global_max_pooling(FEATURE_MAP input, int** index_row, int** index_col);

int*** initial_index_max_pooling(int batch_size, int channels, int size_output_square);

void free_index_max_pooling(int*** index, int batch_size, int channels);

int* argmax_batch(float* arr, int size, int batch_size);
