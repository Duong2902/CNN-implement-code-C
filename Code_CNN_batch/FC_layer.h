#include"stdio.h"
#include"stdlib.h"
#include "feature_map.h" 

typedef struct
{
    int input_dim;
    int output_dim;
    float **weights;
    float *bias;
    float **input;
    float **output;
    float **grad;
} FC_layer;

float **initialize_weight(int input_dim, int output_dim);

float *initialize_bias(int output_dim);

void initialize_FC_layer(FC_layer *layer, float **input, int input_dim, int output_dim, int batch_size, int epoch, int num_train);

void load_FC_layer_weights(FC_layer* layer, const char* filename);

void load_FC_layer_bias(FC_layer* layer, const char* filename);

void free_FC_layer(FC_layer *layer);

void batch_normalize_fc(FC_layer* fc, int batch_size, float* scale, float* shift);

void backprop_FC(float **grad, FC_layer *layer, float learning_rate, int batch_size);

FEATURE_MAP unflatten(float **grad_flatten, int channels, int height, int width, int batch_size);
