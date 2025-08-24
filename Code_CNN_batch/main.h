#include "convolution.h"
#include "backprop.h"
#include "load_file.h"
#include "FC_layer.h"
#include "activation.h"
#include "math.h"

#define INPUT_SIZE 32        // Input: (CIFAR10: 32 x 32 x 3)
#define INPUT_CHANNEL 3   
#define KERNEL_CHANNEL INPUT_CHANNEL
#define KERNEL_SIZE 3
#define FILTER_NUMBER 64
#define NUM_TRAIN 300
#define NUM_VALID 200
#define NUM_TEST 100
#define BATCH_SIZE 32
#define NUM_MINI_BATCH (NUM_TRAIN / BATCH_SIZE)
#define EPOCH 10
#define NUM_CLASSES 10
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9