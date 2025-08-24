#include "main.h"

FEATURE_MAP init_feature_map(int height, int width, int channels, int batch_size)
{
    FEATURE_MAP feature_map;
    feature_map.batch_size = batch_size;
    feature_map.height = height;
    feature_map.width = width;
    feature_map.channels = channels;

    feature_map.fm_value = (float ****)malloc(batch_size * sizeof(float ***));
    for (int b = 0; b < batch_size; b++)
    {
        feature_map.fm_value[b] = (float ***)malloc(channels * sizeof(float **));
        for (int c = 0; c < channels; c++)
        {
            feature_map.fm_value[b][c] = (float **)malloc(height * sizeof(float *));
            for (int h = 0; h < height; h++)
            {
                feature_map.fm_value[b][c][h] = (float *)calloc(width ,sizeof(float));
                
            }
        }
    }

    return feature_map;
}

void initialize_input_values(FEATURE_MAP *input)
{
    for (int b = 0; b < input->batch_size; b++)
    {
        for (int c = 0; c < input->channels; c++)
        {
            for (int i = 0; i < input->height; i++)
            {
                for (int j = 0; j < input->width; j++)
                {
                    input->fm_value[b][c][i][j] = ((float)rand() / RAND_MAX);
                }
            }
        }
    }
}

void print_feature_map_values(const FEATURE_MAP *input, char *str)
{
    printf("%s feature map: \n", str);

    for (int b = 0; b < input->batch_size; b++)
    {
        printf("Number in batch %d:\n", b);
        for (int c = 0; c < input->channels; c++)
        {
            printf("Channel %d:\n", c);
            for (int i = 0; i < input->height; i++)
            {
                for (int j = 0; j < input->width; j++)
                {
                    printf("%.6f ", input->fm_value[b][c][i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

/*
void free_feature_map(FEATURE_MAP *input) {
    for (int channel = 0; channel < input->channels; channel++) {
        for (int h = 0; h < input->height; h++) {
            free(input->fm_value[channel][h]);
        }
        free(input->fm_value[channel]);
    }
    free(input->fm_value);
}
*/

void free_feature_map(FEATURE_MAP *input)
{
    for (int batch_size = 0; batch_size < input->batch_size; batch_size++)
    {
        for (int channel = 0; channel < input->channels; channel++)
        {
            for (int h = 0; h < input->height; h++)
            {
                free(input->fm_value[batch_size][channel][h]);
            }
            free(input->fm_value[batch_size][channel]);
        }
        free(input->fm_value[batch_size]);
    }
    free(input->fm_value);
}

FEATURE_MAP padding_feature_map(FEATURE_MAP input, int padding)
{
    //print_feature_map_values(&input, "input");
    int padding_height = input.height + 2 * padding;
    int padding_width = input.width + 2 * padding;
    FEATURE_MAP ifm_pad = init_feature_map(padding_height, padding_width, input.channels, input.batch_size);

    for (int b = 0; b < ifm_pad.batch_size; b++) 
    {
        for (int c = 0; c < ifm_pad.channels; c++)
        {
            for (int i = 0; i < padding_height; i++)
            {
                for (int j = 0; j < padding_width; j++)
                {
                    ifm_pad.fm_value[b][c][i][j] = 0;
                }
            }
        }
    }

    for (int b = 0; b < ifm_pad.batch_size; b++)
    {
        for (int c = 0; c < input.channels; c++)
        {
            for (int i = 0; i < input.height; i++)
            {
                for (int j = 0; j < input.width; j++)
                {
                    ifm_pad.fm_value[b][c][i + padding][j + padding] = input.fm_value[b][c][i][j];
                }
            }
        }
    }
    return ifm_pad;
}

/*FEATURE_MAP *init_feature_map_array(int num, int height, int width, int channels)
{
    FEATURE_MAP *array = (FEATURE_MAP *)malloc(num * sizeof(FEATURE_MAP));
    for (int i = 0; i < num; i++)
    {
        array[i] = init_feature_map(height, width, channels);
        initialize_input_values(&array[i]);
    }
    return array;
}
*/

/*void free_feature_map_array(FEATURE_MAP *array, int num)
{
    for (int i = 0; i < num; i++)
    {
        free_feature_map(&array[i]);
    }
    free(array);
}
*/
/*FEATURE_MAP **init_feature_map_batch(FEATURE_MAP *feature_map, int batch_size)
{
    if (NUM_TRAIN % batch_size == 0)
    {
        int count = 0;
        int number_mini_batch = NUM_TRAIN / batch_size;
        FEATURE_MAP **feature_maps = malloc(number_mini_batch * sizeof(FEATURE_MAP *));
        for (int i = 0; i < number_mini_batch; i++)
        {
            feature_maps[i] = malloc(batch_size * sizeof(FEATURE_MAP));
        }
        for (int i = 0; i < number_mini_batch; i++)
        {
            for (int j = 0; j < batch_size; j++)
            {
                feature_maps[i][j] = feature_map[count];
                count++;
            }
        }
    }
    else
    {
        int number_mini_batch = NUM_TRAIN / batch_size;
        int last_batch_size = NUM_TRAIN - number_mini_batch * batch_size;
        FEATURE_MAP **feature_maps = malloc(number_mini_batch + 1 * sizeof(FEATURE_MAP *));
        for (int i = 0; i <= number_mini_batch; i++)
        {
            if (i == number_mini_batch)
                feature_maps[i] = malloc(last_batch_size * sizeof(FEATURE_MAP));
            else
                feature_maps[i] = malloc(batch_size * sizeof(FEATURE_MAP));
        }
    }
    return feature_map;
}*/