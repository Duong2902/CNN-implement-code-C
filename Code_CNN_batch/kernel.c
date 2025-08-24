#include "main.h"

KERNEL init_kernel(int size, int channels, int filters)
{
    KERNEL kernel;
    kernel.size = size;
    kernel.channels = channels;
    kernel.filters = filters;
    kernel.kernel_value = (float ****)malloc(filters * sizeof(float ***));
    for (int f = 0; f < filters; f++)
    {
        kernel.kernel_value[f] = (float ***)malloc(channels * sizeof(float **));
        for (int c = 0; c < channels; c++)
        {
            kernel.kernel_value[f][c] = (float **)malloc(size * sizeof(float *));
            for (int s = 0; s < size; s++)
            {
                kernel.kernel_value[f][c][s] = (float *)malloc(size * sizeof(float));
            }
        }
    }
    return kernel;
}

void initialize_kernel_values(KERNEL *kernel)
{
    srand(time(NULL));
    for (int f = 0; f < kernel->filters; f++)
    {
        for (int c = 0; c < kernel->channels; c++)
        {
            for (int i = 0; i < kernel->size; i++)
            {
                for (int j = 0; j < kernel->size; j++)
                {
                    kernel->kernel_value[f][c][i][j] = (float)rand() / RAND_MAX;
                }
            }
        }
    }
}

void print_kernel_kernel_value(const KERNEL *kernel) {
    printf("KERNEL: \n");
    for (int filter = 0; filter < kernel->filters; filter++) {
        printf("Filter %d:\n", filter);
        for (int channel = 0; channel < kernel->channels; channel++) {
            printf("  Channel %d:\n", channel);

            for (int i = 0; i < kernel->size; i++) {
                for (int j = 0; j < kernel->size; j++) {
                    printf(" %.6f ", kernel->kernel_value[filter][channel][i][j]);
                }
                printf("\n"); 
            }
            printf("\n"); 
        }
        printf("\n"); 
    }
}

void load_kernel_from_file(KERNEL* kernel, const char* filename, int size, int channels, int filters) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error: Cannot open kernel file %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int c = 0; c < channels; c++) {
                for (int f = 0; f < filters; f++) {
                    if (fread(&kernel->kernel_value[f][c][i][j], sizeof(float), 1, file) != 1) {
                        printf("Error: Failed to read feature map from file %s\n", filename);
                        fclose(file);
                        exit(1);
                    }
                }
            }
        }
    }
    fclose(file);
}

void free_kernel(KERNEL *kernel)
{
    for (int f = 0; f < kernel->filters; f++)
    {
        for (int c = 0; c < kernel->channels; c++)
        {
            for (int s = 0; s < kernel->size; s++)
            {
                free(kernel->kernel_value[f][c][s]);
            }
            free(kernel->kernel_value[f][c]);
        }
        free(kernel->kernel_value[f]);
    }
    free(kernel->kernel_value);
}

void copy_kernel(KERNEL *dest, KERNEL *src) {
    
    for (int f = 0; f < dest->filters; f++) {
        for (int c = 0; c < dest->channels; c++) {
            for (int i = 0; i < dest->size; i++) {
                for (int j = 0; j < dest->size; j++) {
                    dest->kernel_value[f][c][i][j] = src->kernel_value[f][c][i][j];
                }
            }
        }
    }
}

void copy_bias(float **dest, float *src, int size) {
    for (int i = 0; i < size; i++) {
        (*dest)[i] = src[i];
    }
}