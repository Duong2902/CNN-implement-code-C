#include "main.h"
#include "feature_map.h"

float **initialize_weight(int input_dim, int output_dim)
{
    float **weight = (float **)malloc(input_dim * sizeof(float *));
    for (int i = 0; i < input_dim; i++)
    {
        weight[i] = (float *)calloc(output_dim, sizeof(float));
    }
    return weight;
}

float *initialize_bias(int output_dim)
{
    float *bias = (float *)calloc(output_dim, sizeof(float));
    return bias;
}

void initialize_FC_layer(FC_layer *layer, float **input, int input_dim, int output_dim, int batch_size, int epoch, int num_train)
{
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    layer->input = input;
    if (epoch == 0 && num_train == 0)
    {
        layer->weights = initialize_weight(input_dim, output_dim);
        for (int i = 0; i < input_dim; i++)
        {
            for (int j = 0; j < output_dim; j++)
            {
                layer->weights[i][j] = 1.0; //((float)rand() / RAND_MAX) * 2 - 1
            }
        }

        layer->bias = initialize_bias(output_dim);
        for (int i = 0; i < output_dim; i++)
        {
            layer->bias[i] = 0.0; //((float)rand() / RAND_MAX) * 0.01
        }
        // cấp phát bộ nhớ cho layer->grad
        layer->grad = (float **)malloc(layer->input_dim * sizeof(float *));
        for (int i = 0; i < layer->input_dim; i++)
        {
            layer->grad[i] = (float *)calloc(batch_size, sizeof(float));
        }
        // cấp phát bộ nhớ cho layer->output
        layer->output = (float **)malloc(output_dim * sizeof(float *));
        for (int i = 0; i < output_dim; i++)
        {
            layer->output[i] = (float *)malloc(batch_size * sizeof(float));
        }
    }

    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < output_dim; i++)
        {
            layer->output[i][b] = 0.0f;
            for (int j = 0; j < input_dim; j++)
            {
                layer->output[i][b] += input[j][b] * (layer->weights[j][i]);
            }
            layer->output[i][b] += layer->bias[i];
        }
    }
    for (int i = 0; i < input_dim; i++)
    {
        for (int b = 0; b < batch_size; b++)
        {
            layer->grad[i][b] = 0.0f;
        }
    }
}



void load_FC_layer_weights(FC_layer* layer, const char* filename) {
    int input_dim = layer->input_dim;
    int output_dim = layer->output_dim;
    layer->weights = initialize_weight(input_dim, output_dim);
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Error: Cannot open weight file %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < input_dim; i++) {
        for (int j = 0; j < output_dim; j++) {
            if (fread(&layer->weights[i][j], sizeof(float), 1, fp) != 1) {
                printf("Error: Failed to read weight from file %s\n", filename);
                fclose(fp);
                exit(1);
            }
        }
    }
    fclose(fp);
}

void load_FC_layer_bias(FC_layer* layer, const char* filename) {
    int output_dim = layer->output_dim;
    layer->bias = initialize_bias(output_dim);

    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error: Cannot open bias file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < output_dim; i++) {
        if (fread(&layer->bias[i], sizeof(float), 1, file) != 1) {
            printf("Error: Failed to read bias from file %s\n", filename);
            fclose(file);
            free(layer->bias);
            exit(1);
        }
    }
    fclose(file);
}

/*
void batch_normalize_fc(FC_layer* fc, int batch_size, float* scale, float* shift) {
    float epsilon = 1e-5;
    int output_dim = fc->output_dim;

    if (!fc->output || !scale || !shift) {
        printf("Error: Null pointer in batch_normalize_fc\n");
        return;
    }

    float* mean = (float*)malloc(output_dim * sizeof(float));
    float* variance = (float*)malloc(output_dim * sizeof(float));

    if (!mean || !variance) {
        printf("Error: Memory allocation failed in batch_normalize_fc\n");
        exit(1);
    }

    for (int i = 0; i < output_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            sum += fc->output[j * output_dim + i];
        }
        mean[i] = sum / batch_size;
    }

    for (int i = 0; i < output_dim; i++) {
        float sum_sq = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            float diff = fc->output[j * output_dim + i] - mean[i];
            sum_sq += diff * diff;
        }
        variance[i] = sum_sq / batch_size;
    }

    for (int i = 0; i < output_dim; i++) {
        float stddev = sqrt(variance[i] + epsilon);
        for (int j = 0; j < batch_size; j++) {
            int index = j * output_dim + i;
            fc->output[index] = scale[i] * ((fc->output[index] - mean[i]) / stddev) + shift[i];
        }
    }

    free(mean);
    free(variance);
}
*/


//batch_size=1

void backprop_FC(float **grad, FC_layer *layer, float learning_rate, int batch_size)
{
    float **dw = initialize_weight(layer->input_dim, layer->output_dim);
    float *db = initialize_bias(layer->output_dim);

    // tính dw
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < layer->input_dim; i++)
        {
            for (int j = 0; j < layer->output_dim; j++)
            {
                dw[i][j] += grad[j][b] * layer->input[i][b];
            }
        }
    }
    for (int i = 0; i < layer->input_dim; i++)
    {
        for (int j = 0; j < layer->output_dim; j++)
        {
            dw[i][j] /= batch_size;
        }
    }

    // tính db
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < layer->output_dim; i++)
        {
            db[i] += grad[i][b];
        }
    }
    for (int j = 0; j < layer->output_dim; j++)
    {
        db[j] /= batch_size;
    }

    // tính dx
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < layer->input_dim; i++)
        {
            layer->grad[i][b] = 0.0f;
            for (int j = 0; j < layer->output_dim; j++)
            {
                layer->grad[i][b] += grad[j][b] * layer->weights[i][j];
            }
        }
    }
    // update weights
    for (int i = 0; i < layer->input_dim; i++)
    {
        for (int j = 0; j < layer->output_dim; j++)
        {
            layer->weights[i][j] -= dw[i][j] * learning_rate;
        }
    }

    // update bias
    for (int i = 0; i < layer->output_dim; i++)
    {
        layer->bias[i] -= db[i] * learning_rate;
    }
    // free
    for (int i = 0; i < layer->input_dim; i++)
    {
        free(dw[i]);
    }
    free(dw);
    free(db);
}


FEATURE_MAP unflatten(float **grad_flatten, int channels, int height, int width, int batch_size)
{
    FEATURE_MAP grad_matric = init_feature_map(height, width, channels, batch_size);
    for (int b = 0; b < batch_size; b++)
    {
        size_t k = 0;
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    grad_matric.fm_value[b][c][i][j] = grad_flatten[k++][b];
                }
            }
        }
    }
    return grad_matric;
}


void free_FC_layer(FC_layer *layer)
{
    if (layer == NULL)
        return;

    for (int i = 0; i < layer->input_dim; i++)
    {
        free(layer->weights[i]);
        free(layer->grad[i]);
    }
    for (int j = 0; j < layer->output_dim; j++)
    {
        free(layer->output[j]);
    }
    free(layer->weights);
    free(layer->bias);
    free(layer->output);
    free(layer->grad);
}

