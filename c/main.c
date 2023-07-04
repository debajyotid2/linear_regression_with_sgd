#include <stdio.h>
#include <sys/time.h>
#include <argp.h>
#include "src/matrix.h"
#include "src/helpers.h"
#include "src/losses.h"
#include "src/sgd.h"
#include "src/stats.h"

// Argp argument parser configuration
const char* argp_program_version = "v.0.0.1";
const char* argp_program_bug_address = "<ddebnath@purdue.edu>";
static char doc[] = "A demonstration of linear regression using gradient descent and stochastic gradient descent.";

static struct argp_option options[] = {
    {"n_features", 'N', "N_FEATURES", 0, "Number of features"},
    {"n_samples", 'M', "N_SAMPLES", 0, "Number of samples"},
    {"bias", 'b', "BIAS", OPTION_ARG_OPTIONAL, "Bias term"},
    {"noise intensity", 'I', "NOISE_INTENSITY", OPTION_ARG_OPTIONAL, "Intensity of Gaussian noise to be added"},
    {"learning_rate", 'n', "LEARNING_RATE", OPTION_ARG_OPTIONAL, "Learning rate for the gradient descent"},
    {"batch_size", 'B', "BATCH_SIZE", OPTION_ARG_OPTIONAL, "Batch size"},
    {"seed", 'S', "SEED", OPTION_ARG_OPTIONAL, "Random number seed"},
    {"test_frac", 'f', "TEST_FRAC", OPTION_ARG_OPTIONAL, "Fraction of data for test set"},
    {"n_iter", 'i', "N_ITER", OPTION_ARG_OPTIONAL, "Number of iterations"},
    {"tol", 't', "TOL", OPTION_ARG_OPTIONAL, "Tolerance for convergence"},
    {0}};

// Struct to hold all arguments
struct arguments
{
    unsigned int n_iter;
    double tol;
    unsigned int n_features, n_samples;
    double bias, noise_intensity;
    double learning_rate;
    unsigned int batch_size;
    double test_frac;
    unsigned int seed;
};

// Initialize arguments to defaults
void init_arguments(struct arguments* arg_vals)
{
    arg_vals->n_iter = 10000;
    arg_vals->tol = 0.001;
    arg_vals->n_features = 20; 
    arg_vals->n_samples = 100000;
    arg_vals->bias = -300.7; 
    arg_vals->noise_intensity = 2.0;
    arg_vals->learning_rate = 0.001;
    arg_vals->batch_size = 32;
    arg_vals->test_frac = 0.2;
    arg_vals->seed = 42;
}

// Print arguments
void print_arguments(struct arguments* arg_vals)
{
    printf("Arguments:\n"
           "n_iter = %u, tol = %f,\n"
           "n_features = %u, n_samples = %u\n"
           "bias = %f, noise_intensity = %f\n"
           "learning_rate = %f, batch_size = %u\n"
           "test_frac = %f, seed = %u\n\n",
           arg_vals->n_iter, arg_vals->tol,
           arg_vals->n_features, arg_vals->n_samples,
           arg_vals->bias, arg_vals->noise_intensity,
           arg_vals->learning_rate, arg_vals->batch_size,
           arg_vals->test_frac, arg_vals->seed);
}

// Function to parse arguments option by option
static error_t parse_opt(int key, char* arg, struct argp_state* state)
{
    struct arguments *arguments = (struct arguments*)(state->input);

    switch (key)
    {
        case 'N':
            arguments->n_features = atoi(arg);
            break;
        case 'M':
            arguments->n_samples = atoi(arg);
            break;
        case 'b':
            arguments->bias = atof(arg);
            break;
        case 'I':
            arguments->noise_intensity = atof(arg);
            break;
        case 'n':
            arguments->learning_rate = atof(arg);
            break;
        case 'B':
            arguments->batch_size = atoi(arg);
            break;
        case 'S':
            arguments->seed = atoi(arg);
            break;
        case 'i':
            arguments->n_iter = atoi(arg);
            break;
        case 't':
            arguments->tol = atof(arg);
            break;
        case 'f':
            arguments->test_frac = atof(arg);
            break;
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

// Argument parser
static struct argp argparser = {options, parse_opt, 0, doc};

int main(int argc, char** argv)
{
    struct arguments arg_vals;
    
    // Parse arguments
    init_arguments(&arg_vals);
    argp_parse(&argparser, argc, argv, 0, 0, &arg_vals);
    print_arguments(&arg_vals);

    double duration = 0;

    struct timeval start_t, end_t;
    
    // Generate dataset
    Matrix x = mat_create(arg_vals.n_samples, arg_vals.n_features);
    Matrix y = mat_create(arg_vals.n_samples, 1);
    Matrix x_test = mat_create(arg_vals.n_samples * arg_vals.test_frac, 
                               arg_vals.n_features);
    Matrix y_test = mat_create(arg_vals.n_samples * arg_vals.test_frac, 1);
    Matrix x_train = mat_create(arg_vals.n_samples - x_test.nrows, 
                                arg_vals.n_features);
    Matrix y_train = mat_create(arg_vals.n_samples - x_test.nrows, 1);

    Matrix y_pred;
    
    SGDResult result;
    
    make_regression_dataset(&x, &y, arg_vals.bias, 
                    arg_vals.noise_intensity, arg_vals.seed);
    split_into_train_test(&x, &y, &x_train, &y_train, 
            &x_test, &y_test, arg_vals.seed);
    
    gettimeofday(&start_t, NULL);

    // Gradient descent
    result = gradient_descent(&x_train, &y_train, arg_vals.learning_rate, 
                              &l2_loss, &l2_gradient,
                              arg_vals.n_iter, arg_vals.tol, arg_vals.seed);

    gettimeofday(&end_t, NULL);

    duration = (end_t.tv_sec - start_t.tv_sec) + (end_t.tv_usec - start_t.tv_usec) / 1000000.0;
    
    printf("Gradient descent took %.6f seconds.\n", duration);

    y_pred = mat_mul(&x_test, false, &(result.theta_sol), false);
    mat_add_scalar(&y_pred, result.bias);
    
    printf("MSE: %.4f\n", l2_loss(&y_test, &y_pred));
    printf("MAE: %.4f\n", stats_mae(&y_test, &y_pred));
    printf("R-squared: %.4f\n", stats_r2(&y_test, &y_pred));

    mat_destroy(&y_pred);
    destroy_sgdresult(&result);

    // Stochastic gradient descent
    gettimeofday(&start_t, NULL);
    
    result = stochastic_gradient_descent(&x_train, &y_train, arg_vals.batch_size,
                              arg_vals.learning_rate, &l2_loss, &l2_gradient,
                              arg_vals.n_iter, arg_vals.tol, arg_vals.seed);
    gettimeofday(&end_t, NULL);

    duration = (end_t.tv_sec - start_t.tv_sec) + (end_t.tv_usec - start_t.tv_usec) / 1000000.0;
    printf("Stochastic gradient descent took %.6f seconds.\n", duration);
    
    y_pred = mat_mul(&x_test, false, &(result.theta_sol), false);
    mat_add_scalar(&y_pred, result.bias);

    printf("MSE: %.4f\n", l2_loss(&y_test, &y_pred));
    printf("MAE: %.4f\n", stats_mae(&y_test, &y_pred));
    printf("R-squared: %.4f\n", stats_r2(&y_test, &y_pred));
 
    mat_destroy(&y_pred);
    destroy_sgdresult(&result);
    
    // Destroy dataset
    mat_destroy(&x);
    mat_destroy(&y);
    mat_destroy(&x_train);
    mat_destroy(&y_train);
    mat_destroy(&x_test);
    mat_destroy(&y_test);

    return 0;
}
