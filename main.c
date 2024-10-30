#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TRAINING_SIZE (sizeof(or_train) / sizeof(or_train[0]))

float sigmoidf(float x) {
    return (1.f / (1.f + expf(-x)));
}

typedef float sample[3];

// OR Gate
sample or_train[] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

// AND Gate
sample and_train[] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};

// NAND Gate
sample nand_train[] = {
    {0, 0, 1},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

sample* train = nand_train;

// Works like Javascript rand(), generating a number 0 <= x <= 1
float random_float(void) {
    return ((float) rand() / (float) RAND_MAX);
}

float cost(float w1, float w2, float b) {
    float errResult = 0.0f;
    for (size_t i = 0; i < TRAINING_SIZE; i++) {
        int x1 = train[i][0];
        int x2 = train[i][1];
        float y = train[i][2];
        float yPred = sigmoidf(x2*w2 + x1*w1 + b);
        float d = y - yPred;
        errResult += d*d; // We know error is \sum_{i=1} {N} (y - f(x))^2
        // printf("Predicted: %f, Actual: %f\n", yPred, y);
    }
    errResult /= TRAINING_SIZE; // We want to reduce this metric to 0.
    return errResult;
}

// y = w1x + w0
int main() 
{
    srand(time(0));

    // We randomly initialize our weights and bias
    float w1 = random_float();
    float w2 = random_float();
    float b = random_float();
    // printf("w1: %f, w2: %f\n", w1, w2);
    // printf("Bias: %f\n", b);

    // These are our hyperparameters
    float epsilon = 1e-3;
    float rate = 0.5;


    // Training
    for (size_t i = 0; i < 1000; i++) {
        // These are gradients
        float c = cost(w1, w2, b);
        float dw1 = (cost(w1 + epsilon, w2, b) - c) / epsilon; // Gives us the direction of cost travel, the slope.
        float dw2 = (cost(w1, w2 + epsilon, b) - c) / epsilon;
        float dbias = (cost(w1, w2, b + epsilon) - c) / epsilon;
        w1 -= rate*dw1;
        w2 -= rate*dw2;
        b -= rate*dbias;
        printf("Cost: %f, w1: %f, w2: %f, Bias: %f\n", cost(w1, w2, b), w1, w2, b);
    }

    // Trained model
    printf("w1: %f, w2; %f, Bias: %f\n", w1, w2, b);

    for (size_t i = 0; i < TRAINING_SIZE; i++) {
        int x1 = train[i][0];
        int x2 = train[i][1];
        float y = train[i][2];
        float yPred = sigmoidf(w2*x2 + w1*x1 + b);
        printf("Predicted probability: %f, Actual: %f\n", yPred, y);
    }
    return 0;
}