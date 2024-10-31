#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define TRAINING_SIZE (sizeof(xor_train) / sizeof(xor_train[0]))

typedef struct {
    float w1;
    float w2;
    float b;
} Neuron;

typedef struct {
    Neuron or;
    Neuron nand;
    Neuron and;
} Xor;

typedef float sample[3];

// XOR Gate
sample xor_train[] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

sample* train = xor_train;

float sigmoidf(float x) {
    return (1.f / (1.f + expf(-x)));
}

float forward_propagate(Xor m, float x1, float x2) {
    float a = sigmoidf(m.or.w1 * x1 + m.or.w2 * x2 + m.or.b);
    float b = sigmoidf(m.nand.w1 * x1 + m.nand.w2 * x2 + m.nand.b);
    float result = sigmoidf(m.and.w1 * a + m.and.w2 * b + m.and.b);
    // printf("a: %f, b: %f, result: %f\n", a, b, result);
    return result;
}

float random_float(void) {
    return ((float) rand() / (float) RAND_MAX);
}

// Returns a randomly initialized XOR model
Xor random_init() {
    Xor m;
    m.or.w1 = random_float();
    m.or.w2 = random_float();
    m.or.b = random_float();
    m.nand.w1 = random_float();
    m.nand.w2 = random_float();
    m.nand.b = random_float();
    m.and.w1 = random_float();
    m.and.w2 = random_float();
    m.and.b = random_float();
    return m;
}

void print_model(Xor m) {
    printf("or_w1 = %f\n", m.or.w1);
    printf("or_w2 = %f\n", m.or.w2);
    printf("or_b = %f\n", m.or.b);
    printf("nand_w1 = %f\n", m.nand.w1);
    printf("nand_w2 = %f\n", m.nand.w2);
    printf("nand_b = %f\n", m.nand.b);
    printf("and_w1 = %f\n", m.and.w1);
    printf("and_w2 = %f\n", m.and.w2);
    printf("and_b = %f\n", m.and.b);
}

float cost(Xor m) {
    float errResult = 0.0f;
    for (size_t i = 0; i < TRAINING_SIZE; i++) {
        int x1 = train[i][0];
        int x2 = train[i][1];
        float y = train[i][2];
        float yPred = forward_propagate(m, x1, x2);
        float d = y - yPred;
        errResult += d*d; // We know error is \sum_{i=1} {N} (y - f(x))^2
        // printf("Predicted: %f, Actual: %f\n", yPred, y);
    }
    errResult /= TRAINING_SIZE; // We want to reduce this metric to 0.
    return errResult;
}

// A function that takes in a model, and then calculates the finite differences differential for all values in an XOR model
Xor finite_differences(Xor m, float epsilon) {
    // For each value I need to calculate the dw1, dw2, and db. 
    Xor saved_model = m;
    Xor gradients;
    // We modify the model with a small differential, saving it in a new variable, and then calculating the cost of the new model compared to the previous model. Do this for all variables.
    saved_model.or.w1 += epsilon;
    gradients.or.w1 = (cost(saved_model) - cost(m)) / epsilon;
    saved_model = m;

    saved_model.or.w2 += epsilon;
    gradients.or.w2 = (cost(saved_model) - cost(m)) / epsilon;
    saved_model = m;

    saved_model.or.b += epsilon;
    gradients.or.b = (cost(saved_model) - cost(m)) / epsilon;
    saved_model = m;

    saved_model.nand.w1 += epsilon;
    gradients.nand.w1 = (cost(saved_model) - cost(m)) / epsilon;
    saved_model = m;

    saved_model.nand.w2 += epsilon;
    gradients.nand.w2 = (cost(saved_model) - cost(m)) / epsilon;
    saved_model = m;

    saved_model.nand.b += epsilon;
    gradients.nand.b = (cost(saved_model) - cost(m)) / epsilon;
    saved_model = m;

    saved_model.and.w1 += epsilon;
    gradients.and.w1 = (cost(saved_model) - cost(m)) / epsilon;
    saved_model = m;

    saved_model.and.w2 += epsilon;
    gradients.and.w2 = (cost(saved_model) - cost(m)) / epsilon;
    saved_model = m;

    saved_model.and.b += epsilon;
    gradients.and.b = (cost(saved_model) - cost(m)) / epsilon;
    saved_model = m;

    // We return the model with the finite differences
    return gradients;
}

Xor apply_finite_differences(Xor m, Xor finite_differences, float rate) {
    m.or.w1 -= rate * finite_differences.or.w1;
    m.or.w2 -= rate * finite_differences.or.w2;
    m.or.b -= rate * finite_differences.or.b;
    m.nand.w1 -= rate * finite_differences.nand.w1;
    m.nand.w2 -= rate * finite_differences.nand.w2;
    m.nand.b -= rate * finite_differences.nand.b;
    m.and.w1 -= rate * finite_differences.and.w1;
    m.and.w2 -= rate * finite_differences.and.w2;
    m.and.b -= rate * finite_differences.and.b;
    return m;
}

void test_model(Xor m) {
    for (size_t i = 0; i < TRAINING_SIZE; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = train[i][2];
        float yPred = forward_propagate(m, x1, x2);
        // printf("Predicted: %f, Actual: %f\n", yPred, y);
        printf("%f xor %f = %f, Actual: %f\n", x1, x2, yPred, y);
    }
}

int main(void) {
    srand(time(0));
    Xor m = random_init();
    print_model(m);
    printf("Cost: %f\n", cost(m));

    float eps = 1e-3;
    float rate = 0.5;

    // Gradient descent
    for (int i = 0; i < 100000; i++) {
        float model_cost = cost(m);
        printf("Cost: %f\n", model_cost);
        Xor fd = finite_differences(m, eps);
        m = apply_finite_differences(m, fd, rate);
    }
    test_model(m);
    return 0;
}