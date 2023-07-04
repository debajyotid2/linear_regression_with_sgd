// Tests for module helpers.h
#include "../src/matrix.h"
#include "../src/helpers.h"
#include <catch2/catch_all.hpp>

TEST_CASE("Helpers for linear regression.", "[helpers]")
{
    unsigned int n_features = 20, n_samples = 200;
    unsigned int seed = 2000;
    
    Matrix x = mat_create(n_samples, n_features);
    Matrix y = mat_create(n_samples, 1);
    
    SECTION("Creating linear regression dataset.")
    {
        make_regression_dataset(&x, &y, 1.0, 0.0, seed);
    }

    mat_destroy(&x);
    mat_destroy(&y);
}
