#include "main.h"

float* backprop_CNN_bias(FEATURE_MAP matrix_gradient)
{
    float* bias_grad = (float*)calloc(matrix_gradient.channels, sizeof(float));

    for (int c = 0; c < matrix_gradient.channels; c++)
    {
        for (int b = 0; b < matrix_gradient.batch_size; b++)
        {
            for (int i = 0; i < matrix_gradient.height; i++)
            {
                for (int j = 0; j < matrix_gradient.width; j++)
                {
                    bias_grad[c] += matrix_gradient.fm_value[b][c][i][j];
                }
            }
        }
    }

    return bias_grad;
}

KERNEL backprop_CNN_weight(FEATURE_MAP input, FEATURE_MAP matrix_gradient, KERNEL kernel, int padding, int stride)
{
    input = (padding > 0) ? padding_feature_map(input, padding) : input;
    int weight_grad_size = kernel.size;
    int weight_grad_channels = input.channels;
    int weight_grad_filters = kernel.filters;
    int batch_size = input.batch_size;
    KERNEL output = init_kernel(weight_grad_size, weight_grad_channels, weight_grad_filters);
    float **partial_sum = (float **)malloc((weight_grad_size) * sizeof(float *));
    for (int i = 0; i < weight_grad_size; i++)
    {
        partial_sum[i] = (float *)malloc((weight_grad_size) * sizeof(float));
    }

    for (int f = 0; f < weight_grad_filters; f++)
    {
        for (int c = 0; c < weight_grad_channels; c++)
        {
            for (int i = 0; i < weight_grad_size; i++)
            {
                for (int j = 0; j < weight_grad_size; j++)
                {
                    partial_sum[i][j] = 0.0f;
                }
            }
            for (int b = 0; b < batch_size; b++)
            {
                for (int i = 0; i < matrix_gradient.height; i++)
                {
                    for (int j = 0; j < matrix_gradient.width; j++)
                    {
                        for (int m = 0; m < weight_grad_size; m++)
                        {
                            for (int n = 0; n < weight_grad_size; n++)
                            {
                                // printf("m n: %d %d\n", m, n);
                                partial_sum[m][n] += (input.fm_value[b][c][i * stride + m][j * stride + n] * matrix_gradient.fm_value[b][f][i][j]);
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < weight_grad_size; i++) {
                for (int j = 0; j < weight_grad_size; j++) {
                    output.kernel_value[f][c][i][j] = partial_sum[i][j];
                }
            }
        }
    }
    for (int i = 0; i < weight_grad_size; i++)
    {
        free(partial_sum[i]);
    }
    free(partial_sum);

    if (padding > 0)
    {
        free_feature_map(&input);
    }
    return output;
}

// void update_weight_SGD(KERNEL* kernel, float learning_rate, KERNEL gradient_weight) {
//     for (int f = 0; f < kernel->filters; f++) {
//         for (int c = 0; c < kernel->channels; c++) {
//             for (int i = 0; i < kernel->size; i++) {
//                 for (int j = 0; j < kernel->size; j++) {
//                     kernel->kernel_value[f][c][i][j] -= (learning_rate * gradient_weight.kernel_value[f][c][i][j]);
//                 }
//             }
//         }
//     }
// }

float**** init_v_w(int size, int channels, int filters) {
    // Kiểm tra tham số đầu vào
    if (size <= 0 || channels <= 0 || filters <= 0) {
        printf("Error: Invalid dimensions in init_v_w (size=%d, channels=%d, filters=%d)\n", size, channels, filters);
        return NULL;
    }

    // Cấp phát cấp 1: mảng con trỏ 3 cấp
    float**** v_w = (float****)malloc(filters * sizeof(float***));
    if (v_w == NULL) {
        printf("Error: Failed to allocate v_w\n");
        return NULL;
    }

    // Cấp phát các cấp con trỏ
    for (int f = 0; f < filters; f++) {
        v_w[f] = (float***)malloc(channels * sizeof(float**));
        if (v_w[f] == NULL) {
            // Giải phóng bộ nhớ đã cấp phát
            for (int i = 0; i < f; i++) {
                for (int c = 0; c < channels; c++) {
                    for (int s = 0; s < size; s++) {
                        free(v_w[i][c][s]);
                    }
                    free(v_w[i][c]);
                }
                free(v_w[i]);
            }
            free(v_w);
            printf("Error: Failed to allocate v_w[%d]\n", f);
            return NULL;
        }

        for (int c = 0; c < channels; c++) {
            v_w[f][c] = (float**)malloc(size * sizeof(float*));
            if (v_w[f][c] == NULL) {
                // Giải phóng bộ nhớ đã cấp phát
                for (int i = 0; i < f; i++) {
                    for (int c2 = 0; c2 < channels; c2++) {
                        for (int s = 0; s < size; s++) {
                            free(v_w[i][c2][s]);
                        }
                        free(v_w[i][c2]);
                    }
                    free(v_w[i]);
                }
                for (int c2 = 0; c2 < c; c2++) {
                    for (int s = 0; s < size; s++) {
                        free(v_w[f][c2][s]);
                    }
                    free(v_w[f][c2]);
                }
                free(v_w[f]);
                free(v_w);
                printf("Error: Failed to allocate v_w[%d][%d]\n", f, c);
                return NULL;
            }

            for (int s = 0; s < size; s++) {
                v_w[f][c][s] = (float*)malloc(size * sizeof(float));
                if (v_w[f][c][s] == NULL) {
                    // Giải phóng bộ nhớ đã cấp phát
                    for (int i = 0; i < f; i++) {
                        for (int c2 = 0; c2 < channels; c2++) {
                            for (int s2 = 0; s2 < size; s2++) {
                                free(v_w[i][c2][s2]);
                            }
                            free(v_w[i][c2]);
                        }
                        free(v_w[i]);
                    }
                    for (int c2 = 0; c2 <= c; c2++) {
                        for (int s2 = 0; s2 < (c2 == c ? s : size); s2++) {
                            free(v_w[f][c2][s2]);
                        }
                        free(v_w[f][c2]);
                    }
                    free(v_w[f]);
                    free(v_w);
                    printf("Error: Failed to allocate v_w[%d][%d][%d]\n", f, c, s);
                    return NULL;
                }
                // Khởi tạo giá trị ngay khi cấp phát
                for (int j = 0; j < size; j++) {
                    v_w[f][c][s][j] = 0.0f;
                }
            }
        }
    }
    return v_w;
}

void free_v_w(float**** v_w, int size, int channels, int filters) {
    for (int f = 0; f < filters; f++) {
        for (int c = 0; c < channels; c++) {
            for (int s = 0; s < size; s++) {
                free((v_w)[f][c][s]);
            }
            free((v_w)[f][c]);
        }
        free((v_w)[f]);
    }
    free(v_w);
}

void update_weight_SGD(KERNEL* kernel, KERNEL gradient_weight_avg, float learning_rate, float momentum, float**** v_w) {
    if (kernel == NULL || kernel->kernel_value == NULL) {
        printf("Error: kernel or kernel->kernel_value is NULL\n");
        return;
    }
    if (gradient_weight_avg.kernel_value == NULL) {
        printf("Error: gradient_weight_avg.kernel_value is NULL\n");
        return;
    }
    if (v_w == NULL) {
        printf("Error: v_w is NULL\n");
        return;
    }
    int filters = kernel->filters;
    int channels = kernel->channels;
    int size = kernel->size;
    for (int f = 0; f < filters; f++) {
        if (v_w[f] == NULL || gradient_weight_avg.kernel_value[f] == NULL) {
            printf("Error: v_w[%d] or gradient_weight_avg.kernel_value[%d] is NULL\n", f, f);
            return;
        }
        for (int c = 0; c < channels; c++) {
            if (v_w[f][c] == NULL || gradient_weight_avg.kernel_value[f][c] == NULL) {
                printf("Error: v_w[%d][%d] or gradient_weight_avg.kernel_value[%d][%d] is NULL\n", f, c, f, c);
                return;
            }
            for (int i = 0; i < size; i++) {
                if (v_w[f][c][i] == NULL || gradient_weight_avg.kernel_value[f][c][i] == NULL) {
                    printf("Error: v_w[%d][%d][%d] or gradient_weight_avg.kernel_value[%d][%d][%d] is NULL\n", f, c, i, f, c, i);
                    return;
                }
                for (int j = 0; j < size; j++) {
                    v_w[f][c][i][j] = v_w[f][c][i][j] * momentum + learning_rate * gradient_weight_avg.kernel_value[f][c][i][j];
                    kernel->kernel_value[f][c][i][j] -= v_w[f][c][i][j];
                }
            }
        }
    }
}

float* init_v_b(int filter) {
    float* v_b = (float*)malloc(filter * sizeof(float));
    if (!v_b) {
        fprintf(stderr, "malloc v_b failed\n");
        exit(1);
    }
    for (int i = 0; i < filter; i++) {
        v_b[i] = 0.0f;
    }
    return v_b;
}

void update_bias_SGD(float** bias, float* gradient_bias_avg, FEATURE_MAP output, float learning_rate, float momentum, float* v_b) {
    for (int c = 0; c < output.channels; c++) {
        (v_b)[c] = (v_b)[c] * momentum + learning_rate * gradient_bias_avg[c];
        (*bias)[c] -= (v_b)[c];
    }
}

KERNEL kernel_rotate_180(KERNEL kernel)
{
    int size = kernel.size;
    int channels = kernel.channels;
    int filters = kernel.filters;
    KERNEL output = init_kernel(size, channels, filters);
    for (int f = 0; f < filters; f++)
    {
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    output.kernel_value[f][c][i][j] = kernel.kernel_value[f][c][size - 1 - i][size - 1 - j];
                }
            }
        }
    }
    return output;
}

KERNEL switch_channels_filters(KERNEL kernel)
{
    KERNEL output = init_kernel(kernel.size, kernel.filters, kernel.channels);
    for (int c = 0; c < kernel.channels; c++)
    {
        for (int f = 0; f < kernel.filters; f++)
        {
            for (int i = 0; i < kernel.size; i++)
            {
                for (int j = 0; j < kernel.size; j++)
                {
                    output.kernel_value[c][f][i][j] = kernel.kernel_value[f][c][i][j];
                }
            }
        }
    }
    return output;
}

FEATURE_MAP update_input(FEATURE_MAP matrix_gradient, KERNEL kernel, int stride)
{
    KERNEL rota_kernel = kernel_rotate_180(kernel);
    KERNEL final_kernel = switch_channels_filters(rota_kernel);
    free_kernel(&rota_kernel);
    int padding = final_kernel.size - 1;
    float *bias = (float *)calloc(final_kernel.filters, sizeof(float));

    FEATURE_MAP output = Conv2D(matrix_gradient, final_kernel, padding, stride, bias); // với stride = 1
    free_kernel(&final_kernel);
    free(bias);
    return output;
}

FEATURE_MAP backprop_max_pooling(FEATURE_MAP input, FEATURE_MAP gradient_matrix, int kernel_size, int stride, int*** index_row, int*** index_col)
{
    int out_height = input.height;
    int out_width = input.width;
    int out_channels = input.channels;
    int out_batch_size = input.batch_size;
    int slide_count;
    int slide_row_count;
    int slide_col_count;
    FEATURE_MAP output = init_feature_map(out_height, out_width, out_channels, out_batch_size);
    for (int b = 0; b < out_batch_size; b++)
    {
        for (int c = 0; c < out_channels; c++)
        {
            slide_count = 0;
            slide_row_count = 0;
            for (int i = 0; i < out_height; i += stride)
            {
                slide_col_count = 0;
                for (int j = 0; j < out_width; j += stride)
                {
                    if (slide_count >= 16 * 16) {
                        printf("slide_count overflow! b=%d c=%d\n", b, c);
                        exit(1);
                    }
                    for (int m = 0; m < kernel_size; m++)
                    {
                        for (int n = 0; n < kernel_size; n++)
                        {
                            if ((i + m == index_row[b][c][slide_count]) && (j + n == index_col[b][c][slide_count]))
                            {
                                output.fm_value[b][c][i + m][j + n] = gradient_matrix.fm_value[b][c][slide_row_count][slide_col_count];
                                // printf("slide_col slide_row: %d %d\n", slide_col_count, slide_row_count);
                            }
                        }
                    }
                    slide_count++;
                    slide_col_count++;
                }
                slide_row_count++;
            }
        }
    }
    return output;
}

/*FEATURE_MAP backprop_global_max_pooling(FEATURE_MAP input, float* matrix_gradient, int* index_row, int* index_col) {
    int out_height = input.height;
    int out_width = input.width;
    int out_channels = input.channels;
    FEATURE_MAP output = init_feature_map(out_height, out_width, out_channels);
    for (int c = 0; c < out_channels; c++) {
        for (int i = 0; i < out_height; i++) {
            for (int j = 0; j < out_width; j++) {
                if (i == index_row[c] && j == index_col[c]) output.fm_value[c][i][j] = matrix_gradient[c];
                else output.fm_value[c][i][j] = 0;
            }
        }
    }
    return output;
}
*/

float **backprop_loss_softmax(int* y, float* y_predict, int number_of_category, int batch_size)
{
    float **grad = (float **)malloc(number_of_category * sizeof(float *));
    for (int i = 0; i < number_of_category; i++)
    {
        grad[i] = (float *)calloc(batch_size, sizeof(float));
    }
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < number_of_category; i++)
        {
            grad[i][b] = (y_predict[b * number_of_category + i] - (float)y[b * number_of_category + i]) / batch_size;
        }
    }
    return grad;
}

void free_backprop_loss_softmax(float **loss_softmax, int number_of_category){
    for (int i =0; i< number_of_category; i++){
        free(loss_softmax[i]);
    }
    free(loss_softmax);
}


void kernel_gradient_average(KERNEL* kernel_gradient, int batch_size) {
    for (int f = 0; f < kernel_gradient->filters; f++) {
        for (int c = 0; c < kernel_gradient->channels; c++) {
            for (int i = 0; i < kernel_gradient->size; i++) {
                for (int j = 0; j < kernel_gradient->size; j++) {
                    kernel_gradient->kernel_value[f][c][i][j] = kernel_gradient->kernel_value[f][c][i][j] / batch_size;
                }
            }
        }
    }
}

void bias_gradient_average(float** bias_gradient, int channels, int batch_size) {
    for (int c = 0; c < channels; c++) {
        (*bias_gradient)[c] = (*bias_gradient)[c] / batch_size;
    }
}