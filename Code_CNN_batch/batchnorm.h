#include "feature_map.h"

typedef struct
{
    int channels;
    float *gamma;
    float *beta;
    float *grad_gamma;
    float *grad_beta;
} BatchNormLayer;

void init_batchnorm_layer(BatchNormLayer *bn_layer, int channels);

FEATURE_MAP batchnorm_forward(FEATURE_MAP input, BatchNormLayer* bn_layer);

FEATURE_MAP batchnorm_backward(FEATURE_MAP d_y, FEATURE_MAP input, BatchNormLayer *bn_layer, float learning_rate);

void free_batchnorm_layer(BatchNormLayer* bn_layer);
