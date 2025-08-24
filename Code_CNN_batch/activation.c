#include "main.h"

float relu(float x) {
    return (x > 0) ? x : 0;
}

float* softmax(float** input, int number_of_category, int batch_size)
{
    float* y = (float*)malloc(number_of_category*batch_size * sizeof(float));
    
    if (y == NULL)
    {
        return NULL;
    }
    float* max_val = (float*)calloc(batch_size, sizeof(float));
    if (max_val == NULL)
    {
        return NULL;
    }
    for (int b = 0; b < batch_size; b++)
    {
        max_val[b] = input[0][b];
        for (int i = 1; i < number_of_category; i++)
        {
            if (input[i][b] > max_val[b])
            {
                max_val[b] = input[i][b];
            }
        }
    }
    for (int b = 0; b < batch_size; b++)
    {
        float sum = 0.0f;
        for (int i = 0; i <number_of_category; i++)
        {
            y[b*number_of_category+i] = expf(input[i][b] - max_val[b]);
            sum += y[b*number_of_category+i];
        }
        for (int i = 0; i < number_of_category; i++)
        {
            y[b*number_of_category+i] /= sum;
        }
    }
    free(max_val);
    return y;
}

void free_softmax(float*y)
{
    free(y);
}


float cross_entropy_loss(int *y, float *y_predict, int number_of_category, int batch_size)
{
    float L = 0.0f;
    const float epsilon = 1e-5f;
    for (int b = 0; b < batch_size; b++)
    {
        //printf("num: %d\n", *num);
        for (int i = 0; i < number_of_category; i++)
        {
            L += -((float)y[b * number_of_category + i] * logf(y_predict[b * number_of_category + i]+epsilon));
        }
    }
    return L / (float)batch_size;
}