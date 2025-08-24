#include "main.h"

void load_feature_map_from_bin(FEATURE_MAP* fm, const char* filename, int number) {
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Error: Cannot open feature map file %s\n", filename);
        exit(1);
    }
    //printf("file: %s\n", filename);
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            for (int c = 0; c < INPUT_CHANNEL; c++) {
                //printf("c: %d\n", c);
                //printf("number c i j: %d %d %d %d", number, c, i, j);
                if (fread(&fm->fm_value[number][c][i][j], sizeof(float), 1, fp) != 1) {
                    printf("Error: Failed to read feature map from file %s\n", filename);
                    fclose(fp);
                    exit(1);
                }
                //printf("fm[%d][%d][%d][%d]: %f\n", number, c, i, j, fm->fm_value[number][c][i][j]);
            }
        }
    }
    fclose(fp);
}

int* load_label_from_txt(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Fail to open label file");
        exit(EXIT_FAILURE);
    }

    int* label_row = (int*)malloc(NUM_CLASSES * sizeof(int));
    if (label_row == NULL) {
        perror("Fail to allocate memory");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
        if (fscanf(fp, "%d", &label_row[i]) != 1) {
            fprintf(stderr, "Error label %d từ %s\n", i, filename);
            fclose(fp);
            free(label_row);
            exit(EXIT_FAILURE);
        }
    }

    fclose(fp);
    return label_row;
}

int** load_all_train_labels(int num_samples, const char* label_folder) { 
    int** labels = (int**)malloc(num_samples * sizeof(int*));  // mình đang cần trải phảng hết ra giống như trong hàm sofmax và crossentrop
    if (labels == NULL) {
        perror("Fail to alocate memory");
        exit(EXIT_FAILURE);
    }

    char filename[256];

    for (int i = 0; i < num_samples; i++) {
        sprintf(filename, "%slabel_train_%d.txt", label_folder, i);
        //printf("%s\n", filename);
        labels[i] = load_label_from_txt(filename);
        //printf("Đã load label: %s\n", filename);
    }

    return labels;
}

int** load_all_valid_labels(int num_samples, const char* label_folder) {
    int** labels = (int**)malloc(num_samples * sizeof(int*));  // tại sao lại cấp phát mảng 2 chiều => chưa cấp phát chiều còn lại=> mình đang cần trải phảng hết ra giống như trong hàm sofmax và crossentropy
    if (labels == NULL) {
        perror("Fail to alocate memory");
        exit(EXIT_FAILURE);
    }

    char filename[256];

    for (int i = 0; i < num_samples; i++) {
        sprintf(filename,  "%slabel_train_%d.txt", label_folder, i+NUM_TRAIN);
        //printf("%s\n", filename);
        labels[i] = load_label_from_txt(filename);
        //printf("Đã load label: %s\n", filename);
    }

    return labels;
}

int** load_all_test_labels(int num_samples, const char* label_folder) {
    int** labels = (int**)malloc(num_samples * sizeof(int*)); 
    if (labels == NULL) {
        perror("Fail to alocate memory");
        exit(EXIT_FAILURE);
    }

    char filename[256];

    for (int i = 0; i < num_samples; i++) {
        sprintf(filename,  "%slabel_test_%d.txt", label_folder, i);
        //printf("%s\n", filename);
        labels[i] = load_label_from_txt(filename);
        //printf("Đã load label: %s\n", filename);
    }

    return labels;
}

void load_all_train_data(FEATURE_MAP* train_samples, const char* img_folder, int* count, int batch_size) {
    char filename[256];
    (*train_samples) = init_feature_map(32, 32, 3, batch_size);
    for (int i = 0; i < batch_size; i++) {
        //printf("i: %d\n", i);
        sprintf(filename, "%strain_img_%d.bin", img_folder, (*count));
        load_feature_map_from_bin(train_samples, filename, i);
        (*count)++;
    }
}

void load_all_train_data_res(FEATURE_MAP* train_samples, const char* img_folder, int* count) {
    char filename[256];
    int res = NUM_TRAIN - NUM_MINI_BATCH * BATCH_SIZE;
    (*train_samples) = init_feature_map(32, 32, 3, res);
    for (int i = 0; i < res; i++) {
        //printf("i: %d\n", i);
        sprintf(filename, "%strain_img_%d.bin", img_folder, (*count));
        load_feature_map_from_bin(train_samples, filename, i);
        (*count)++;
    }
}

void load_all_test_data(FEATURE_MAP* test_samples, const char* img_folder, int* count, int batch_size) {
    char filename[256];
    (*test_samples) = init_feature_map(32, 32, 3, 1);
    for (int i = 0; i < 1; i++) {
        //printf("i: %d\n", i);
        sprintf(filename, "%stest_img_%d.bin", img_folder, (*count));
        load_feature_map_from_bin(test_samples, filename, i);
        (*count)++;
    }
}

float* load_m_v_s_s_conv(const char* filename, int channels) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error: Cannot open scale_shift file %s\n", filename);
        exit(1);
    }
    float* scale_shift = (float*)malloc(channels * sizeof(float));
    if (scale_shift == NULL) {
        printf("Error: Memory allocation for scale_shift failed\n");
        fclose(file);
        exit(1);
    }
    for (int i = 0; i < channels; i++) {
        if (fread(&scale_shift[i], sizeof(float), 1, file) != 1) {
            printf("Error: Failed to read scale_shift from file %s\n", filename);
            fclose(file);
            free(scale_shift);
            exit(1);
        }
    }
    fclose(file);
    return scale_shift;
}

float* load_bias(const char* filename, int filters) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error: Cannot open bias file %s\n", filename);
        exit(1);
    }
    float* bias = (float*)malloc(filters * sizeof(float));
    if (bias == NULL) {
        printf("Error: Memory allocation for bias failed\n");
        fclose(file);
        exit(1);
    }
    for (int i = 0; i < filters; i++) {
        if (fread(&bias[i], sizeof(float), 1, file) != 1) {
            printf("Error: Failed to read bias from file %s\n", filename);
            fclose(file);
            free(bias);
            exit(1);
        }
    }
    fclose(file);
    return bias;
}

int *flatten_label(int **load_label, int num_sample, int num_of_category){
    int *y= (int*) malloc(num_sample*num_of_category*sizeof(int));

    if(load_label==NULL){
        return NULL;
    }

    for(int i=0; i<num_sample; i++){
        for(int j=0; j<num_of_category; j++){
            y[i*num_of_category+j]= load_label[i][j];
        }
    }
    return y;
}

void free_label(int **label_2d, int num_sample){
    for(int i=0; i< num_sample; i++){
        free(label_2d[i]);
    }
    free(label_2d);
}
