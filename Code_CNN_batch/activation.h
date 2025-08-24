#include "stdio.h"

float relu(float x);

float* softmax(float** input, int number_of_category, int batch_size);

void free_softmax(float* y);

float cross_entropy_loss(int* y, float* y_predict, int number_of_category, int batch_size);
