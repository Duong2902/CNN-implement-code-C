#include "main.h"
float *initialize_bias_conv2d(int channels)
{
    float *bias = (float *)malloc(channels * sizeof(float));
    for (int i = 0; i < channels; i++)
    {
        bias[i] = ((float)rand() / RAND_MAX) * 0.01;
    }
    return bias;
}

FEATURE_MAP Conv2D(FEATURE_MAP input, KERNEL kernel, int padding, int stride, float *bias)
{
    input = (padding > 0) ? padding_feature_map(input, padding) : input;
    int out_height = (input.height - kernel.size) / stride + 1;
    int out_width = (input.width - kernel.size) / stride + 1;

    FEATURE_MAP output = init_feature_map(out_height, out_width, kernel.filters, input.batch_size);
    for (int b = 0; b < output.batch_size; b++)
    {
        for (int f = 0; f < kernel.filters; f++)
        {
            for (int i = 0; i < out_height; i++)
            {
                for (int j = 0; j < out_width; j++)
                {
                    float sum = 0.0f;
                    for (int c = 0; c < kernel.channels; c++)
                    {
                        for (int m = 0; m < kernel.size; m++)
                        {
                            for (int n = 0; n < kernel.size; n++)
                            {
                                sum += (input.fm_value[b][c][i * stride + m][j * stride + n] * kernel.kernel_value[f][c][m][n]);
                            }
                        }
                    }
                    output.fm_value[b][f][i][j] = sum + bias[f];
                }
            }
        }
    }
    if (padding > 0)
    {
        free_feature_map(&input);
    }
    return output;
}

FEATURE_MAP max_pooling(FEATURE_MAP input, int kernel_size, int stride, int ****index_row, int ****index_col)
{
    int out_height = (input.height - kernel_size) / stride + 1;
    int out_width = (input.width - kernel_size) / stride + 1;
    FEATURE_MAP output = init_feature_map(out_height, out_width, input.channels, input.batch_size);

    int max_slide_count = out_height * out_width;
    if (max_slide_count > 16 * 16) {
        fprintf(stderr, "Error: Output size (%d) exceeds allocated index size (%d)\n", 
                max_slide_count, 16 * 16);
        exit(1);
    } 

    *index_row = initial_index_max_pooling(input.batch_size, input.channels, max_slide_count);
    *index_col = initial_index_max_pooling(input.batch_size, input.channels, max_slide_count);

    //printf("batch_size: %d\n", input.batch_size);

    for (int b = 0; b < input.batch_size; b++)
    {
        for (int c = 0; c < input.channels; c++)
        {
            int slide_count = 0;
            for (int i = 0; i < out_height; i++)
            {
                for (int j = 0; j < out_width; j++)
                {
                    float max = input.fm_value[b][c][i * stride][j * stride];
                    (*index_row)[b][c][slide_count] = i * stride;
                    (*index_col)[b][c][slide_count] = j * stride;

                    for (int m = 0; m < kernel_size; m++)
                    {
                        for (int n = 0; n < kernel_size; n++)
                        {
                            int row = i * stride + m;
                            int col = j * stride + n;
                            if (input.fm_value[b][c][row][col] > max)
                            {
                                max = input.fm_value[b][c][row][col];
                                (*index_row)[b][c][slide_count] = row;
                                (*index_col)[b][c][slide_count] = col;
                            }
                        }
                    }

                    output.fm_value[b][c][i][j] = max;
                    slide_count++;
                }
            }
        }
    }
    return output;
}

/*FEATURE_MAP avg_pooling(FEATURE_MAP input, int kernel_size, int stride) {
    int out_height = (input.height - kernel_size) / stride + 1;
    int out_width = (input.width - kernel_size) / stride + 1;
    FEATURE_MAP output = init_feature_map(out_height, out_width, input.channels);
    float max = input.fm_value[0][0][0];
    for (int c = 0; c < input.channels; c++) {
        for (int i = 0; i < out_height; i++) {
            for (int j = 0; j < out_width; j++) {
                float sum = 0;
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        sum += input.fm_value[c][i * stride + m][j * stride + n];
                    }
                }
                output.fm_value[c][i][j] = sum / (kernel_size * kernel_size);
            }
        }
    }
    return output;
}*/

/*float* global_max_pooling(FEATURE_MAP input, int** index_row, int** index_col) {
    int channels = input.channels;
    int height = input.height;
    int width = input.width;
    float* result = (float*)malloc(channels * sizeof(float));
    for (int c = 0; c < channels; c++) {
        float max = input.fm_value[c][0][0];
        (*index_row)[c] = 0;
        (*index_col)[c] = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (max < input.fm_value[c][i][j]) {
                    max = input.fm_value[c][i][j];
                    (*index_row)[c] = i;
                    (*index_col)[c] = j;
                }
            }
        }
        result[c] = max;
    }
    return result;
}
*/


float **Flatten(FEATURE_MAP input, int flat_dim)
{
    float **output = (float **)malloc(flat_dim * sizeof(float *));
    for (int i = 0; i < flat_dim; i++)
    {
        output[i] = (float *)malloc(input.batch_size * sizeof(float));
    }
    for (int b = 0; b < input.batch_size; b++)
    {
        size_t k = 0;
        for (int c = 0; c < input.channels; c++)
        {
            for (int h = 0; h < input.height; h++)
            {
                for (int w = 0; w < input.width; w++)
                {
                    output[k][b] = input.fm_value[b][c][h][w];
                    k++;
                }
            }
        }
    }
    return output;
}
void free_Flatten(float **input, int flat_dim)
{
    for (int i = 0; i < flat_dim; i++)
    {
        free(input[i]);
    }
    free(input);
}

/*int argmax(float* arr, int length) {
    int idx = 0;
    float max = arr[0];
    for (int i = 1; i < length; i++) {
        if (arr[i] > max) {
            max = arr[i];
            idx = i;
        }
    }
    return idx;
}
*/
int*** initial_index_max_pooling(int batch_size, int channels, int size_output_square)
{
    int*** index = (int***)malloc(batch_size * sizeof(int**));
    if (!index) {
        printf("malloc index failed\n");
        exit(1);
    }

    for (int b = 0; b < batch_size; b++)
    {
        index[b] = (int**)malloc(channels * sizeof(int*));
        if (!index[b]) {
            printf("malloc index[%d] failed\n", b);
            exit(1);
        }

        for (int c = 0; c < channels; c++)
        {
            index[b][c] = (int*)malloc(size_output_square * sizeof(int));
            if (!index[b][c]) {
                printf("malloc index[%d][%d] failed\n", b, c);
                exit(1);
            }
        }

    }

    return index;
}


/*int* initial_index_global_max_pooling(int channels) {
    int* index = (int*)malloc(channels * sizeof(int));
    return index;
}
*/
void free_index_max_pooling(int*** index, int batch_size, int channels)
{
    if (!index) return;
    for (int b = 0; b < batch_size; b++)
    {
        if (!index[b]) continue;
        for (int c = 0; c < channels; c++)
        {
            free(index[b][c]);
            index[b][c] = NULL;
        }
        free(index[b]);
        index[b] = NULL;
    }
    free(index);
}

int* argmax_batch(float* arr, int size, int batch_size) {
    int* idxs = malloc(batch_size * sizeof(int));
    if (!idxs) return NULL;

    for (int b = 0; b < batch_size; b++) {
        float max_val = arr[b*size +0];
        int best = 0;
        for (int i = 1; i < size; i++) {
            if (arr[b*size +i] > max_val) {
                max_val = arr[b*size +i];
                best = i;
            }
        }
        idxs[b] = best;
    }
    return idxs;
}