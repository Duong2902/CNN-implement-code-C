#include "batchnorm.h"

void init_batchnorm_layer(BatchNormLayer *bn_layer, int channels)
{
    bn_layer->channels = channels;
    bn_layer->gamma = (float *)malloc(channels * sizeof(float));
    bn_layer->beta = (float *)malloc(channels * sizeof(float));
    bn_layer->grad_gamma = (float *)malloc(channels * sizeof(float));
    bn_layer->grad_beta = (float *)malloc(channels * sizeof(float));
    if (!bn_layer->gamma || !bn_layer->beta || !bn_layer->grad_gamma || !bn_layer->grad_beta)
    {
        free(bn_layer->gamma);
        free(bn_layer->beta);
        free(bn_layer->grad_gamma);
        free(bn_layer->grad_beta);
        bn_layer->gamma = bn_layer->beta = bn_layer->grad_gamma = bn_layer->grad_beta = NULL;
        return;
    }

    for (int i = 0; i < channels; i++)
    {
        bn_layer->gamma[i] = 1.0f;
        bn_layer->beta[i] = 0.0f;
    }
}
FEATURE_MAP batchnorm_forward(FEATURE_MAP input, BatchNormLayer *bn_layer)
{
    float e = 1e-5;
    int C = input.channels;
    int H = input.height;
    int W = input.width;
    int B = input.batch_size;
    int N = B * H * W;
    float *mean = (float *)calloc(C, sizeof(float));
    float *var = (float *)calloc(C, sizeof(float));

    FEATURE_MAP output = init_feature_map(H, W, C, B);
    for (int i = 0; i < C; i++)
    {
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                for (int k = 0; k < W; k++)
                {
                    mean[i] += input.fm_value[b][i][j][k];
                }
            }
        }
        // tính trung bình
        mean[i] /= N;
    }
    for (int i = 0; i < C; i++)
    {
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                for (int k = 0; k < W; k++)
                {
                    float diff = input.fm_value[b][i][j][k] - mean[i];
                    var[i] += diff * diff;
                }
            }
        }
        // Tính phương sai
        var[i] /= N;
    }
    for (int i = 0; i < C; i++)
    {
        float stddev_inv = 1.0f / sqrtf(var[i] + e);
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                for (int k = 0; k < W; k++)
                {
                    float x_hat = (input.fm_value[b][i][j][k] - mean[i]) * stddev_inv;
                    output.fm_value[b][i][j][k] = bn_layer->gamma[i] * x_hat + bn_layer->beta[i];
                }
            }
        }
    }

    free(mean);
    free(var);
    return output;
}
FEATURE_MAP batchnorm_backward(FEATURE_MAP d_y, FEATURE_MAP input, BatchNormLayer *bn_layer, float learning_rate)
{
    float e = 1e-5;
    int C = input.channels;
    int H = input.height;
    int W = input.width;
    int B = input.batch_size;
    int N = B * H * W;

    float *mean = (float *)calloc(C, sizeof(float));
    float *var = (float *)calloc(C, sizeof(float));
    float *grad_gamma_bn = (float *)calloc(C, sizeof(float));
    float *grad_beta_bn = (float *)calloc(C, sizeof(float));

    for (int i = 0; i < C; i++)
    {
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                for (int k = 0; k < W; k++)
                {
                    mean[i] += input.fm_value[b][i][j][k];
                }
            }
        }
        // tính trung bình
        mean[i] /= N;
    }
    for (int i = 0; i < C; i++)
    {
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                for (int k = 0; k < W; k++)
                {
                    float diff = input.fm_value[b][i][j][k] - mean[i];
                    var[i] += diff * diff;
                }
            }
        }
        // Tính phương sai
        var[i] /= N;
    }

    FEATURE_MAP d_x = init_feature_map(H, W, C, B);

    for (int i = 0; i < C; i++)
    {
        float stddev_inv = 1.0f / sqrtf(var[i] + e);

        float grad_gamma = 0.0f;
        float grad_beta = 0.0f;
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                for (int k = 0; k < W; k++)
                {
                    float x_hat = (input.fm_value[b][i][j][k] - mean[i]) * stddev_inv;
                    grad_gamma += d_y.fm_value[b][i][j][k] * x_hat;
                    grad_beta += d_y.fm_value[b][i][j][k];
                }
            }
        }
        grad_gamma_bn[i] = grad_gamma;
        grad_beta_bn[i] = grad_beta;
        // Tính d_x
        float d_var = 0.0f;
        float d_mean = 0.0f;
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                for (int k = 0; k < W; k++)
                {
                    float x_hat = (input.fm_value[b][i][j][k] - mean[i]) * stddev_inv;
                    float d_xhat = d_y.fm_value[b][i][j][k] * bn_layer->gamma[i];
                    d_var += -0.5f * d_xhat * (input.fm_value[b][i][j][k] - mean[i]) * powf(var[i] + e, -1.5f);
                }
            }
        }
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                for (int k = 0; k < W; k++)
                {
                    float x_hat = (input.fm_value[b][i][j][k] - mean[i]) * stddev_inv;
                    float d_xhat = d_y.fm_value[b][i][j][k] * bn_layer->gamma[i];
                    d_mean += -d_xhat * stddev_inv + -2.0f * d_var * (input.fm_value[b][i][j][k] - mean[i]) / N;
                }
            }
        }
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                for (int k = 0; k < W; k++)
                {
                    float x_hat = (input.fm_value[b][i][j][k] - mean[i]) * stddev_inv;
                    float d_xhat = d_y.fm_value[b][i][j][k] * bn_layer->gamma[i];
                    d_x.fm_value[b][i][j][k] = d_xhat * stddev_inv + d_var * 2.0f * (input.fm_value[b][i][j][k] - mean[i]) / N + d_mean / N;
                }
            }
        }
        // update gamma, beta
        bn_layer->gamma[i] -= learning_rate * (grad_gamma_bn[i] / N);
        bn_layer->beta[i] -= learning_rate * (grad_beta_bn[i] / N);
    }

    free(mean);
    free(var);
    free(grad_beta_bn);
    free(grad_gamma_bn);
    return d_x;
}

void free_batchnorm_layer(BatchNormLayer *bn_layer)
{
    if (bn_layer == NULL)
    {
        return;
    }
    if (bn_layer->gamma != NULL)
    {
        free(bn_layer->gamma);
    }
    if (bn_layer->beta != NULL)
    {
        free(bn_layer->beta);
    }
    if (bn_layer->grad_gamma != NULL)
    {
        free(bn_layer->grad_gamma);
    }
    if (bn_layer->grad_beta != NULL)
    {
        free(bn_layer->grad_beta);
    }
    bn_layer->gamma = bn_layer->beta = bn_layer->grad_gamma = bn_layer->grad_beta = NULL;
}