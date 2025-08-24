#include "kernel.h"
#include "feature_map.h" 

float* backprop_CNN_bias(FEATURE_MAP matrix_gradient);

KERNEL backprop_CNN_weight(FEATURE_MAP input, FEATURE_MAP matrix_gradient, KERNEL kernel, int padding, int stride);

// void update_weight_SGD(KERNEL* kernel, float learning_rate, KERNEL gradient_weight);

float**** init_v_w(int size, int channels, int filters);

void free_v_w(float**** v_w, int size, int channels, int filters);

void update_weight_SGD(KERNEL* kernel, KERNEL gradient_weight_avg, float learning_rate, float momentum, float**** v_w);

float* init_v_b(int channels);

void update_bias_SGD(float** bias, float* gradient_bias_avg, FEATURE_MAP output, float learning_rate, float momentum, float* v_b);

KERNEL kernel_rotate_180(KERNEL kernel);

KERNEL switch_channels_filters(KERNEL kernel);

FEATURE_MAP update_input(FEATURE_MAP matrix_gradient, KERNEL kernel, int stride);

FEATURE_MAP backprop_max_pooling(FEATURE_MAP input, FEATURE_MAP gradient_matrix, int kernel_size, int stride, int ***index_row, int ***index_col);

//FEATURE_MAP backprop_global_max_pooling(FEATURE_MAP input, float* matrix_gradient, int* index_row, int* index_col);

float** backprop_loss_softmax(int* y, float* y_predict, int number_of_category, int batch_size);
void free_backprop_loss_softmax(float **loss_softmax, int number_of_category);

void kernel_gradient_average(KERNEL* kernel_gradient, int batch_size);

void bias_gradient_average(float** bias_gradient, int channels, int batch_size);