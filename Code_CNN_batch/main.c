#include "main.h"

// int main() {
// 	FEATURE_MAP input = init_feature_map(3, 3, 1);
// 	initialize_input_values(&input);
// 	print_feature_map_values(&input, "input");
// 	KERNEL kernel = init_kernel(2, 2, 1);
// 	FEATURE_MAP matrix_gradient = init_feature_map(2, 2, 1);
// 	initialize_input_values(&matrix_gradient);
// 	print_feature_map_values(&matrix_gradient, "matrix_gradient");
// 	KERNEL matrix_kernel = backprop_CNN_weight(input, matrix_gradient, kernel, 0, 1);
// 	print_kernel_values(&matrix_kernel);
// 	return 0;
// }

int main()
{
    printf("Program started!\n");
    int num_mini_batch = NUM_MINI_BATCH;
    
    FEATURE_MAP *train_samples = (FEATURE_MAP *)malloc(num_mini_batch * sizeof(FEATURE_MAP));
    FEATURE_MAP *valid_samples = (FEATURE_MAP *)malloc(NUM_VALID * sizeof(FEATURE_MAP));
    //FEATURE_MAP *test_samples = (FEATURE_MAP *)malloc(NUM_TEST * sizeof(FEATURE_MAP));
    int count;

    count = 0;
    int batch_size = 0;
    for (int i = 0; i < num_mini_batch; i++)
    {
        if (i == num_mini_batch - 1 && NUM_TRAIN % BATCH_SIZE != 0)
            batch_size = BATCH_SIZE;//NUM_TRAIN - (num_mini_batch-1)* BATCH_SIZE;
        else
            batch_size = BATCH_SIZE;

        load_all_train_data(&train_samples[i], "/home/haiduong/EDABK/EDABK_project1_CNN/Code_CNN_ver4/Data_CNN/train/", &count, batch_size);
    }
    count = count+NUM_TRAIN-BATCH_SIZE*num_mini_batch;
    for (int i = 0; i < NUM_VALID; i++) 
    {

        load_all_train_data(&valid_samples[i], "/home/haiduong/EDABK/EDABK_project1_CNN/Code_CNN_ver4/Data_CNN/train/", &count, 1);
    }
    count = 0;
    

    int **train_labels_2d = load_all_train_labels(num_mini_batch*BATCH_SIZE, "/home/haiduong/EDABK/EDABK_project1_CNN/Code_CNN_ver4/Data_CNN/label/label_train/");
    int **valid_labels_2d = load_all_valid_labels(NUM_VALID, "/home/haiduong/EDABK/EDABK_project1_CNN/Code_CNN_ver4/Data_CNN/label/label_train/");
    int **test_labels_2d  = load_all_test_labels  (NUM_TEST, "/home/haiduong/EDABK/EDABK_project1_CNN/Code_CNN_ver4/Data_CNN/label/label_test/");

    int *train_label = flatten_label(train_labels_2d, num_mini_batch*BATCH_SIZE, NUM_CLASSES);
    int *test_label  = flatten_label(test_labels_2d, NUM_TEST, NUM_CLASSES);
    int *valid_label = flatten_label(valid_labels_2d, NUM_VALID, NUM_CLASSES);

    free_label(train_labels_2d,num_mini_batch*BATCH_SIZE);
    free_label(valid_labels_2d,NUM_VALID);
    free_label(test_labels_2d,NUM_TEST);


    KERNEL kernel1 = init_kernel(3, 3, 16);
    KERNEL kernel2 = init_kernel(1, 16, 10);

    // checkpoint
    KERNEL kernel_best1 = init_kernel(3, 3, 16);
    KERNEL kernel_best2 = init_kernel(1, 16, 10);

    initialize_kernel_values(&kernel1);
    initialize_kernel_values(&kernel2);

    float *bias1 = initialize_bias_conv2d(16);
    float *bias2 = initialize_bias_conv2d(10);

    // checkpoint
    float *bias_best1 = initialize_bias_conv2d(16);
    float *bias_best2 = initialize_bias_conv2d(10);

    FC_layer fc;
    // print_feature_map_values(&train_samples[0], "train_samples_0");

    float ****v_w1 = init_v_w(kernel1.size, kernel1.channels, kernel1.filters);
    if (v_w1 == NULL)
    {
        printf("Failed to initialize v_w1\n");
        return 1;
    }
    float *v_b1 = init_v_b(kernel1.filters);

    float ****v_w2 = init_v_w(kernel2.size, kernel2.channels, kernel2.filters);
    if (v_w2 == NULL)
    {
        printf("Failed to initialize v_w2\n");
        return 1;
    }
    float *v_b2 = init_v_b(kernel2.filters);

    batch_size = 0;

    float accuracy;
    float new_acc;

    for (int e = 0; e < 1; e++)
    {
        for (int i = 0; i < num_mini_batch; i++)
        {
            if (i == num_mini_batch - 1 && NUM_TRAIN % BATCH_SIZE != 0)
                batch_size = BATCH_SIZE;//UM_TRAIN - (NUM_MINI_BATCH -1)* BATCH_SIZE;
            else
                batch_size = BATCH_SIZE;
            FEATURE_MAP output_conv1 = Conv2D(train_samples[i], kernel1, 1, 1, bias1);
            int ***index_row1 = NULL;
            int ***index_col1 = NULL;
            FEATURE_MAP output_max_pooling1 = max_pooling(output_conv1, 2, 2, &index_row1, &index_col1);

            FEATURE_MAP output_conv2 = Conv2D(output_max_pooling1, kernel2, 0, 1, bias2);

            size_t input_dim = output_conv2.channels * output_conv2.height * output_conv2.width;

            float **flat_input = Flatten(output_conv2, input_dim);

            initialize_FC_layer(&fc, flat_input, input_dim, 10, batch_size, e, i);

            float *output_softmax = softmax(fc.output, 10, batch_size);

            int* batch_train_label = train_label + i * BATCH_SIZE * NUM_CLASSES; // chia nhỏ train_label theo các mini_batch theo địa chỉ offset

            float **gradient_loss_softmax = backprop_loss_softmax(batch_train_label, output_softmax, 10, batch_size);  

           
            FEATURE_MAP grad_matrix_fc = unflatten(fc.grad, output_conv2.channels, output_conv2.height, output_conv2.width, batch_size);


            KERNEL weight_grad2 = backprop_CNN_weight(output_max_pooling1, grad_matrix_fc, kernel2, 0, 1);

            kernel_gradient_average(&kernel2, batch_size);

            update_weight_SGD(&kernel2, weight_grad2, LEARNING_RATE, MOMENTUM, v_w2);

            float *bias_grad2 = backprop_CNN_bias(grad_matrix_fc);

            bias_gradient_average(&bias_grad2, output_conv2.channels, batch_size);

            update_bias_SGD(&bias2, bias_grad2, output_conv2, LEARNING_RATE, MOMENTUM, v_b2);

            FEATURE_MAP output_max_pooling1_grad = update_input(grad_matrix_fc, kernel2, 1);

            FEATURE_MAP output_conv1_grad = backprop_max_pooling(output_conv1, output_max_pooling1_grad, 2, 2, index_row1, index_col1);
            
            KERNEL weight_grad1 = backprop_CNN_weight(train_samples[i], output_conv1_grad, kernel1, 1, 1);
            
            kernel_gradient_average(&kernel1, batch_size);

            update_weight_SGD(&kernel1, weight_grad1, LEARNING_RATE, MOMENTUM, v_w1);

            float *bias_grad1 = backprop_CNN_bias(output_conv1_grad);

            bias_gradient_average(&bias_grad1, output_conv1.channels, batch_size);

            update_bias_SGD(&bias1, bias_grad1, output_conv1, LEARNING_RATE, MOMENTUM, v_b1);
            
            free_Flatten(flat_input, input_dim);
            free_feature_map(&output_conv2);
            free_feature_map(&output_max_pooling1);
            free_feature_map(&output_conv1);
            free_feature_map(&grad_matrix_fc);
            free_feature_map(&output_max_pooling1_grad);
            free_feature_map(&output_conv1_grad);
            free_kernel(&weight_grad1);
            free_kernel(&weight_grad2);
            free_index_max_pooling(index_row1, batch_size, 16);
            free_index_max_pooling(index_col1, batch_size, 16);
            free(output_softmax);
            free_backprop_loss_softmax(gradient_loss_softmax, NUM_CLASSES);
            free(bias_grad1);
            free(bias_grad2);
        }

        float loss_total = 0;
        int correct = 0;
        for (int i = 0; i < NUM_VALID; i++)
        {
            batch_size =1;
            FEATURE_MAP output_conv1 = Conv2D(valid_samples[i], kernel1, 1, 1, bias1);
            int ***index_row1 = NULL;
            int ***index_col1 = NULL;
            FEATURE_MAP output_max_pooling1 = max_pooling(output_conv1, 2, 2, &index_row1, &index_col1);

            FEATURE_MAP output_conv2 = Conv2D(output_max_pooling1, kernel2, 0, 1, bias2);

            size_t input_dim = output_conv2.channels * output_conv2.height * output_conv2.width;

            float **flat_input = Flatten(output_conv2, input_dim);

            initialize_FC_layer(&fc, flat_input, input_dim, 10, batch_size, e, 1 ); 
            float *output_softmax = softmax(fc.output, 10, batch_size);
            int* batch_valid_label = valid_label + i * batch_size * NUM_CLASSES; // chia nhỏ train_label theo các mini_batch theo địa chỉ offset

            float loss = cross_entropy_loss(batch_valid_label, output_softmax, 10, 1);

            loss_total += loss;

            int *predicted = argmax_batch(output_softmax, 10, batch_size);

            int *true_label = argmax_batch((float *)batch_valid_label, 10, batch_size);

            for (int b = 0; b < batch_size; b++)
            {
                if (predicted[b] == true_label[b])
                {
                    correct++;
                }
            }
            free_index_max_pooling(index_row1, batch_size, 16);
            free_index_max_pooling(index_col1, batch_size, 16);
            free_softmax(output_softmax);
            free(predicted);
            free(true_label);
            free_Flatten(flat_input,input_dim);
            free_feature_map(&output_conv2);
            free_feature_map(&output_max_pooling1);
            free_feature_map(&output_conv1);
        }
        if (e == 0)
            accuracy = 0.0f;
        new_acc = (float)correct / NUM_VALID * 100.0f;
        if (new_acc > accuracy)
        {
            copy_kernel(&kernel_best1, &kernel1);
            copy_kernel(&kernel_best2, &kernel2);
            copy_bias(&bias_best1, bias1, 16);
            copy_bias(&bias_best2, bias2, 10);
            printf("Accuracy improves from %f to %f\n", accuracy, new_acc);
            accuracy = new_acc;
        }
        printf("Test Correct: %d/%d \n", correct, NUM_VALID);
        printf("Test Accuracy: %.2f%%\n", accuracy);

        printf("finish epoch %d\n", e);
    }

    free_v_w(v_w2, kernel2.size, kernel2.channels, kernel2.filters);
    free(v_b2);

    free_v_w(v_w1, kernel1.size, kernel1.channels, kernel1.filters);
    free(v_b1);

    free_FC_layer(&fc);
    free_kernel(&kernel1);
    free_kernel(&kernel2);
    free_kernel(&kernel_best1);
    free_kernel(&kernel_best2);
    free(bias1);
    free(bias2);
    free(bias_best1);
    free(bias_best2);


    for (int i = 0; i < num_mini_batch; i++)
    {
        printf("i: %d", i);
        free_feature_map(&train_samples[i]);
    }
    free(train_samples);
    /*for (int i = 0; i < NUM_TEST; i++)
    {
        free_feature_map(&test_samples[i]);
    }
    free(test_samples);*/
    for (int i = 0; i < NUM_VALID; i++)
    {
        free_feature_map(&valid_samples[i]);
    }
    free(valid_samples);
    free(train_label);
    free(test_label);
    free(valid_label);

}