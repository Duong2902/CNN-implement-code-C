#include "feature_map.h"
#include "kernel.h"

void load_feature_map_from_bin(FEATURE_MAP* fm, const char* filename, int number);

int* load_label_from_txt(const char* filename);

int** load_all_train_labels(int num_samples, const char* label_folder);

int** load_all_valid_labels(int num_samples, const char* label_folder);

int** load_all_test_labels(int num_samples, const char* label_folder);

void free_label(int **label_2d, int num_sample);

/// @brief 
/// @param train_samples 
/// @param img_folder 
/// @param count 
/// @param batch_size 
void load_all_train_data(FEATURE_MAP* train_samples, const char* img_folder, int* count, int batch_size);

void load_all_train_data_res(FEATURE_MAP* train_samples, const char* img_folder, int* count);

void load_all_test_data(FEATURE_MAP* test_samples, const char* img_folder, int* count, int batch_size);

float* load_m_v_s_s_conv(const char* filename, int channels);

float* load_bias(const char* filename, int filters);

int *flatten_label(int **load_label, int num_sample, int num_of_category);