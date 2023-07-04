// Tests for module losses.h

#include <catch2/catch_all.hpp>
#include "../src/matrix.h"
#include "../src/losses.h"

TEST_CASE("L2 loss.", "[losses]")
{
    unsigned int seed = 2234;
    Matrix x = mat_create(20, 10);
    Matrix y = mat_create(20, 1);
    Matrix theta = mat_create(10, 1);
    
    // Fill x, y, theta with data
    mat_fill_random(&x, seed);
    mat_scale(&x, 10.0);
    mat_fill_random(&y, seed);
    mat_scale(&y, -2.5);
    mat_fill_random(&theta, seed);
    mat_scale(&theta, 7.5); 

    SECTION("L2 loss of a vector with itself is zero.")
    {
        mat_fill(&y, 1.0);
        double loss = l2_loss(&y, &y);
        REQUIRE(loss == 0.0);
    }

    SECTION("L2 loss between two different vectors must be as expected.")
    {
        Matrix y_pred = mat_create(20, 1);
        mat_fill_random(&y_pred, seed);

        double loss = l2_loss(&y, &y_pred);
        double loss_exp = 0.0;
        
        for (size_t i=0; i<y.nrows; i++)
            for (size_t j=0; j<y.ncols; j++)
            {
                double diff = y.data[i*y.ncols+j]-y_pred.data[i*y.ncols+j];
                loss_exp += diff * diff;
            }
        loss_exp /= (double)(y.nrows*y.ncols);
        REQUIRE(loss==loss_exp);

        mat_destroy(&y_pred);
    }

    SECTION("Gradient of L2 loss must be correct.")
    {
        // Gradient of L2 loss w.r.t. theta is X^T(X theta - y)
        Matrix x_theta_prod, grad_expect;
        Matrix grad;
        
        grad = l2_gradient(&x, &y, &theta);
        
        x_theta_prod = mat_mul(&x, false, &theta, false);
        mat_sub(&x_theta_prod, &y);
        grad_expect = mat_mul(&x, true, &x_theta_prod, false);

        for (size_t i=0; i<grad_expect.nrows; i++)
            for (size_t j=0; j<grad_expect.ncols; j++)
                REQUIRE(grad.data[i*grad_expect.ncols+j]==\
                        grad_expect.data[i*grad_expect.ncols+j]);
        
        mat_destroy(&x_theta_prod);
        mat_destroy(&grad_expect);
        mat_destroy(&grad);
    }

    mat_destroy(&x);
    mat_destroy(&y);
    mat_destroy(&theta);
}
