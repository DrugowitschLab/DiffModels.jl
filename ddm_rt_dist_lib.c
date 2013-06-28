/**
 * ddm_rt_dist_lib.c - Functions that compute the reaction time distributions
 *                     of drift-diffusion models.
 *
 * 2012-02-28 Jan Drugowitsch   initial release v0.1
 **/

#include "ddm_rt_dist_lib.h"

#include <math.h>
#include <stdlib.h>
#include <assert.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define PI 3.14159265358979323846
#define PI_2 2.0 * PI
#define SERIES_ACC 1e-25


/** ddm_rt_dist_full - compute reaction time distribution
 * mu - vector of drift rates
 * sig2 - vector of diffusion variances
 * b_lo - vector of lower bounds
 * b_up - vector of upper bounds
 * b_lo_deriv - vector of derivative of lower bound
 * b_up_deriv - vector of derivative of upper bound
 * delta_t - step size in [s]
 * k_max - number of steps, t_max = k_max * delta_t
 * 
 * Results are stored in
 * g1 - vector (size k_max) for pdf of fpt, upper boundary
 * g2 - vector (size k_max) for pdf of fpt, lower boundary
 * 
 * All vector are of size k_max, in steps of delta_t
 * 
 * 0 is returned on success, -1 of memory allocation failure
 **/
int ddm_rt_dist_full(double mu[], double sig2[], double b_lo[], double b_up[],
                     double b_lo_deriv[], double b_up_deriv[],
                     double delta_t, int k_max, double g1[], double g2[])
{
    double curr_cum_mu, curr_cum_sig2, sqrt_2_pi, delta_t_sqrt_2_pi;
    double *cum_mu, *cum_sig2;
    int j, k;
    
    assert(mu != NULL && sig2 != NULL && sig2 != NULL & b_lo != NULL &&
           b_up != NULL && b_lo_deriv != NULL && b_up_deriv != NULL &&
           delta_t > 0 && k_max > 0 && g1 != NULL && g2 != NULL);
    
    /* allocate array memory */
    cum_mu = malloc(k_max * sizeof(double));
    cum_sig2 = malloc(k_max * sizeof(double));
    if (cum_mu == NULL || cum_sig2 == NULL) {
        free(cum_mu);
        free(cum_sig2);
        return -1;
    }
    
    /* precompute some constants */
    sqrt_2_pi = 1 / sqrt(2 * PI);
    delta_t_sqrt_2_pi = delta_t * sqrt_2_pi;
    
    /* cumulative mu and sig2 */
    curr_cum_mu = delta_t * mu[0];
    cum_mu[0] = curr_cum_mu;
    curr_cum_sig2 = delta_t * sig2[0];
    cum_sig2[0] = curr_cum_sig2;
    for (j = 1; j < k_max; ++j) {
        curr_cum_mu += delta_t * mu[j];
        cum_mu[j] = curr_cum_mu;
        curr_cum_sig2 += delta_t * sig2[j];
        cum_sig2[j] = curr_cum_sig2;
    }
    
    /* fill up g1 and g2 recursively */
    for (k = 0; k < k_max; ++k) {
        /* speed increase by reducing array access */
        double sig2_k = sig2[k];
        double b_up_k = b_up[k];
        double b_lo_k = b_lo[k];
        double cum_mu_k = cum_mu[k];
        double cum_sig2_k = cum_sig2[k];
        double sqrt_cum_sig2_k = sqrt(cum_sig2_k);
        double b_up_deriv_k = b_up_deriv[k] - mu[k];
        double b_lo_deriv_k = b_lo_deriv[k] - mu[k];
        
        /* initial values */
        double g1_k = -sqrt_2_pi / sqrt_cum_sig2_k * 
                      exp(-0.5 * (b_up_k - cum_mu_k) * (b_up_k - cum_mu_k) / 
                          cum_sig2_k) *
                      (b_up_deriv_k - sig2_k * (b_up_k - cum_mu_k) / cum_sig2_k);
        double g2_k = sqrt_2_pi / sqrt_cum_sig2_k *
                      exp(-0.5 * (b_lo_k - cum_mu_k) * (b_lo_k - cum_mu_k) /
                          cum_sig2_k) *
                      (b_lo_deriv_k - sig2_k * (b_lo_k - cum_mu_k) / cum_sig2_k);
        /* relation to previous values */
        for (j = 0; j < k; ++j) {
            /* reducing array access + pre-compute values */
            double cum_sig2_diff_j = cum_sig2_k - cum_sig2[j];
            double sqrt_cum_sig2_diff_j = sqrt(cum_sig2_diff_j);
            double cum_mu_diff_j = cum_mu[j] - cum_mu_k;
            double b_up_k_up_j_diff = b_up_k - b_up[j] + cum_mu_diff_j;
            double b_up_k_lo_j_diff = b_up_k - b_lo[j] + cum_mu_diff_j;
            double b_lo_k_up_j_diff = b_lo_k - b_up[j] + cum_mu_diff_j;
            double b_lo_k_lo_j_diff = b_lo_k - b_lo[j] + cum_mu_diff_j;
            /* add values */
            g1_k += delta_t_sqrt_2_pi / sqrt_cum_sig2_diff_j *
                    (g1[j] * exp(-0.5 * b_up_k_up_j_diff * b_up_k_up_j_diff / 
                                 cum_sig2_diff_j) *
                     (b_up_deriv_k - 
                      sig2_k * b_up_k_up_j_diff / cum_sig2_diff_j) +
                     g2[j] * exp(-0.5 * b_up_k_lo_j_diff * b_up_k_lo_j_diff /
                                 cum_sig2_diff_j) *
                     (b_up_deriv_k - 
                      sig2_k * b_up_k_lo_j_diff / cum_sig2_diff_j));
            g2_k -= delta_t_sqrt_2_pi / sqrt_cum_sig2_diff_j *
                    (g1[j] * exp(-0.5 * b_lo_k_up_j_diff * b_lo_k_up_j_diff /
                                 cum_sig2_diff_j) *
                     (b_lo_deriv_k -
                      sig2_k * b_lo_k_up_j_diff / cum_sig2_diff_j) +
                     g2[j] * exp(-0.5 * b_lo_k_lo_j_diff * b_lo_k_lo_j_diff /
                                 cum_sig2_diff_j) *
                     (b_lo_deriv_k -
                      sig2_k * b_lo_k_lo_j_diff / cum_sig2_diff_j));
        }
        /* avoid negative densities that could appear due to numerical instab. */
        g1[k] = MAX(g1_k, 0);
        g2[k] = MAX(g2_k, 0);
    }
    
    free(cum_mu);
    free(cum_sig2);
    return 0;
}


/** ddm_rt_dist_full_leak - compute reaction time distribution with leak
 * mu - vector of drift rates
 * sig2 - vector of diffusion variances
 * b_lo - vector of lower bounds
 * b_up - vector of upper bounds
 * b_lo_deriv - vector of derivative of lower bound
 * b_up_deriv - vector of derivative of upper bound
 * inv_leak - 1 / leak time constant
 * delta_t - step size in [s]
 * k_max - number of steps, t_max = k_max * delta_t
 * 
 * Results are stored in
 * g1 - vector (size k_max) for pdf of fpt, upper boundary
 * g2 - vector (size k_max) for pdf of fpt, lower boundary
 * 
 * All vector are of size k_max, in steps of delta_t
 * 
 * 0 is returned on success, -1 of memory allocation failure
 **/
int ddm_rt_dist_full_leak(double mu[], double sig2[],
                          double b_lo[], double b_up[],
                          double b_lo_deriv[], double b_up_deriv[],
                          double inv_leak, double delta_t, int k_max,
                          double g1[], double g2[])
{
    double curr_cum_mu, curr_cum_sig2, curr_disc;
    double sqrt_2_pi, delta_t_sqrt_2_pi, exp_leak, exp2_leak;
    double *cum_mu, *cum_sig2, *disc, *disc2;
    int j, k;
    
    assert(mu != NULL && sig2 != NULL && sig2 != NULL & b_lo != NULL &&
           b_up != NULL && b_lo_deriv != NULL && b_up_deriv != NULL &&
           inv_leak >= 0 &&  delta_t > 0 && k_max > 0 &&
           g1 != NULL && g2 != NULL);
    
    /* allocate array memory */
    cum_mu = malloc(k_max * sizeof(double));
    cum_sig2 = malloc(k_max * sizeof(double));
    disc = malloc(k_max * sizeof(double));
    disc2 = malloc(k_max * sizeof(double));
    if (cum_mu == NULL || cum_sig2 == NULL) {
        free(cum_mu);
        free(cum_sig2);
        free(disc);
        free(disc2);
        return -1;
    }
    
    /* precompute some constants */
    sqrt_2_pi = 1 / sqrt(2 * PI);
    delta_t_sqrt_2_pi = delta_t * sqrt_2_pi;
    exp_leak = exp(- delta_t * inv_leak);
    exp2_leak = exp(- 2 * delta_t * inv_leak);
    
    /* cumulative mu and sig2, and discount (leak) */
    curr_cum_mu = delta_t * mu[0];
    cum_mu[0] = curr_cum_mu;
    curr_cum_sig2 = delta_t * sig2[0];
    cum_sig2[0] = curr_cum_sig2;
    curr_disc = exp_leak;
    disc[0] = curr_disc;
    for (j = 1; j < k_max; ++j) {
        curr_cum_mu = exp_leak * curr_cum_mu + delta_t * mu[j];
        cum_mu[j] = curr_cum_mu;
        curr_cum_sig2 = exp2_leak * curr_cum_sig2 + delta_t * sig2[j];
        cum_sig2[j] = curr_cum_sig2;
        curr_disc *= exp_leak;
        disc[j] = curr_disc;
    }
    /* double discount (leak),
     * note that  disc[k - j - 1] = exp(- inv_leak delta_t (k - j))
     *           disc2[k - j - 1] = exp(- 2 inv_leak delta_t (k - j))
     * such that half of disc can be used to compute disc2 */
    k = (int) floor(((double) (k_max - 1)) / 2);
    for (j = 0; j <= k; ++j)
        disc2[j] = disc[2 * j + 1];
    curr_disc = disc2[k];
    for (j = k + 1; j < k_max; ++j) {
        curr_disc *= exp2_leak;
        disc2[j] = curr_disc;
    }
    
    /* fill up g1 and g2 recursively */
    for (k = 0; k < k_max; ++k) {
        /* speed increase by reducing array access */
        double sig2_k = sig2[k];
        double b_up_k = b_up[k];
        double b_lo_k = b_lo[k];
        double cum_mu_k = cum_mu[k];
        double cum_sig2_k = cum_sig2[k];
        double sqrt_cum_sig2_k = sqrt(cum_sig2_k);
        double b_up_deriv_k = b_up_deriv[k] + inv_leak * b_up_k - mu[k];
        double b_lo_deriv_k = b_lo_deriv[k] + inv_leak * b_lo_k - mu[k];
        
        /* initial values */
        double g1_k = -sqrt_2_pi / sqrt_cum_sig2_k * 
                      exp(-0.5 * (b_up_k - cum_mu_k) * (b_up_k - cum_mu_k) / 
                          cum_sig2_k) *
                      (b_up_deriv_k - sig2_k * (b_up_k - cum_mu_k) / cum_sig2_k);
        double g2_k = sqrt_2_pi / sqrt_cum_sig2_k *
                      exp(-0.5 * (b_lo_k - cum_mu_k) * (b_lo_k - cum_mu_k) /
                          cum_sig2_k) *
                      (b_lo_deriv_k - sig2_k * (b_lo_k - cum_mu_k) / cum_sig2_k);
        /* relation to previous values */
        for (j = 0; j < k; ++j) {
            /* reducing array access + pre-compute values */
            double disc_j = disc[k - j - 1];
            double cum_sig2_diff_j = cum_sig2_k - disc2[k - j - 1] * cum_sig2[j];
            double sqrt_cum_sig2_diff_j = sqrt(cum_sig2_diff_j);
            double cum_mu_diff_j = disc_j * cum_mu[j] - cum_mu_k;
            double b_up_k_up_j_diff = b_up_k - disc_j * b_up[j] + cum_mu_diff_j;
            double b_up_k_lo_j_diff = b_up_k - disc_j * b_lo[j] + cum_mu_diff_j;
            double b_lo_k_up_j_diff = b_lo_k - disc_j * b_up[j] + cum_mu_diff_j;
            double b_lo_k_lo_j_diff = b_lo_k - disc_j * b_lo[j] + cum_mu_diff_j;
            /* add values */
            g1_k += delta_t_sqrt_2_pi / sqrt_cum_sig2_diff_j *
                    (g1[j] * exp(-0.5 * b_up_k_up_j_diff * b_up_k_up_j_diff / 
                                 cum_sig2_diff_j) *
                     (b_up_deriv_k - 
                      sig2_k * b_up_k_up_j_diff / cum_sig2_diff_j) +
                     g2[j] * exp(-0.5 * b_up_k_lo_j_diff * b_up_k_lo_j_diff /
                                 cum_sig2_diff_j) *
                     (b_up_deriv_k - 
                      sig2_k * b_up_k_lo_j_diff / cum_sig2_diff_j));
            g2_k -= delta_t_sqrt_2_pi / sqrt_cum_sig2_diff_j *
                    (g1[j] * exp(-0.5 * b_lo_k_up_j_diff * b_lo_k_up_j_diff /
                                 cum_sig2_diff_j) *
                     (b_lo_deriv_k -
                      sig2_k * b_lo_k_up_j_diff / cum_sig2_diff_j) +
                     g2[j] * exp(-0.5 * b_lo_k_lo_j_diff * b_lo_k_lo_j_diff /
                                 cum_sig2_diff_j) *
                     (b_lo_deriv_k -
                      sig2_k * b_lo_k_lo_j_diff / cum_sig2_diff_j));
        }
        /* avoid negative densities that could appear due to numerical instab. */
        g1[k] = MAX(g1_k, 0);
        g2[k] = MAX(g2_k, 0);
    }
    
    free(cum_mu);
    free(cum_sig2);
    free(disc);
    free(disc2);
    return 0;
}


/** ddm_rt_dist - compute the reaction time distribution
 * mu - vector of drift rates, of size k_max, in steps of delta_t
 * bound - vector of bound heights, of size k_max, in steps of delta_t
 * delta_t - step size in [s]
 * k_max - number of steps, t_max = k_max * delta_t
 * Results are stored in
 * g1 - vector (size k_max) for pdf of first passage times, upper boundary
 * g2 - vector (size k_max) for pdf of first passage times, lower boundary
 *
 * 0 is returned on success, -1 on memory allocation failure
 */
int ddm_rt_dist(double mu[], double bound[], double delta_t, int k_max,
             double g1[], double g2[])
{
    double delta_t_2, pi_delta_t_2, curr_cum_mu;
    double *cum_mu, *bound_deriv, *norm_sqrt_t, *norm_t;
    int j, k;

    assert(mu != NULL && bound != NULL && delta_t > 0 && k_max > 0 &&
           g1 != NULL && g2 != NULL);
    
    /* allocate array memory */
    cum_mu = malloc(k_max * sizeof(double));
    bound_deriv = malloc(k_max * sizeof(double));
    norm_sqrt_t = malloc(k_max * sizeof(double));
    norm_t = malloc(k_max * sizeof(double));
    if (cum_mu == NULL || bound_deriv == NULL || 
        norm_sqrt_t == NULL || norm_t == NULL) {
        free(cum_mu);
        free(bound_deriv);
        free(norm_sqrt_t);
        free(norm_t);
        return -1;
    }

    /* precompute some constants */
    delta_t_2 = 2.0 * delta_t;
    pi_delta_t_2 = PI * delta_t_2;
    
    /* cumulative mu, and derivative of bound */
    curr_cum_mu = delta_t * mu[0];
    cum_mu[0] = curr_cum_mu;
    for (j = 1; j < k_max; ++j) {
        curr_cum_mu += delta_t * mu[j];
        cum_mu[j] = curr_cum_mu;
        bound_deriv[j - 1] = (bound[j] - bound[j - 1]) / delta_t;
    }
    bound_deriv[k_max - 1] = bound_deriv[k_max - 2];

    /* norm_sqrt_t[i] = 1 / sqrt(2 * pi * delta_t * (i + 1)) 
       norm_t[i] = 1 / (delta_t * (i + 1)) */
    for (j = 0; j < k_max; ++j) {
        norm_sqrt_t[j] = 1.0 / sqrt(pi_delta_t_2 * (j + 1.0));
        norm_t[j] = 1.0 / (delta_t * (j + 1.0));
    }

    /* fill up g1 and g2 recursively */
    for (k = 0; k < k_max; ++k) {
        /* speed increase by reducing array access */
        double bound_k = bound[k];
        double bound_deriv_k1 = bound_deriv[k] - mu[k];
        double bound_deriv_k2 = -bound_deriv[k] - mu[k];
        double cum_mu_k = cum_mu[k];
        double norm_t_j = norm_t[k];
        double norm_sqrt_t_j = norm_sqrt_t[k];
        /* initial values */
        double g1_k = -norm_sqrt_t_j
                      * exp(- 0.5 * (bound_k - cum_mu_k) * (bound_k - cum_mu_k) * norm_t_j)
                      * (bound_deriv_k1 - (bound_k - cum_mu_k) * norm_t_j);
        double g2_k = norm_sqrt_t_j
                      * exp(- 0.5 * (-bound_k - cum_mu_k) * (-bound_k - cum_mu_k) * norm_t_j)
                      * (bound_deriv_k2 - (-bound_k - cum_mu_k) * norm_t_j);
        /* relation to previous values */
        for (j = 0; j < k; ++j) {
            /* reducing array access + pre-compute values */
            double bound_j = bound[j];
            double cum_mu_k_j = cum_mu_k - cum_mu[j];
            double diff1 = bound_k - bound_j - cum_mu_k_j;
            double diff2 = bound_k + bound_j - cum_mu_k_j;
            norm_t_j = norm_t[k - j - 1];
            norm_sqrt_t_j = norm_sqrt_t[k - j - 1];
            /* add values */
            g1_k += delta_t * norm_sqrt_t_j
                    * (g1[j] * exp(- 0.5 * diff1 * diff1 * norm_t_j)
                       * (bound_deriv_k1 - diff1 * norm_t_j)
                       + g2[j] * exp(- 0.5 * diff2 * diff2 * norm_t_j)
                       * (bound_deriv_k1 - diff2 * norm_t_j));
            diff1 = -bound_k - bound_j - cum_mu_k_j;
            diff2 = -bound_k + bound_j - cum_mu_k_j;
            g2_k -= delta_t * norm_sqrt_t_j
                    * (g1[j] * exp(- 0.5 * diff1 * diff1 * norm_t_j)
                       * (bound_deriv_k2 - diff1 * norm_t_j)
                       + g2[j] * exp(- 0.5 * diff2 * diff2 * norm_t_j)
                       * (bound_deriv_k2 - diff2 * norm_t_j));
        }
        /* avoid negative densities that could appear due to numerical instab. */
        g1[k] = MAX(g1_k, 0);
        g2[k] = MAX(g2_k, 0);
    }

    free(cum_mu);
    free(bound_deriv);
    free(norm_sqrt_t);
    free(norm_t);
    return 0;
}


/** ddm_rt_dist_const_mu - compute the reaction time distribution, constant mu
 * mu - drift rate
 * bound - vector of bound heights, of size k_max, in steps of delta_t
 * delta_t - step size in [s]
 * k_max - number of steps, t_max = k_max * delta_t
 * Results are stored in
 * g1 - vector (size k_max) for pdf of first passage times, upper boundary
 * g2 - vector (size k_max) for pdf of first passage times, lower boundary
 *
 * 0 is returned on success, -1 on memory allocation failure
 */
int ddm_rt_dist_const_mu(double mu, double bound[], double delta_t, int k_max,
             double g1[], double g2[])
{
    double delta_t_2, pi_delta_t_2, mu_delta_t, mu_2;
    double *bound_deriv, *norm_sqrt_t, *norm_t;
    int j, k;
    
    assert(mu > 0 && bound != NULL && delta_t > 0 && k_max > 0 &&
           g1 != NULL && g2 != NULL);
    
    /* allocate array memory */
    bound_deriv = malloc(k_max * sizeof(double));
    norm_sqrt_t = malloc(k_max * sizeof(double));
    norm_t = malloc(k_max * sizeof(double));
    if (bound_deriv == NULL || norm_sqrt_t == NULL || norm_t == NULL) {
        free(bound_deriv);
        free(norm_sqrt_t);
        free(norm_t);
        return -1;
    }

    /* precompute some constants */
    delta_t_2 = 2.0 * delta_t;
    pi_delta_t_2 = PI * delta_t_2;
    mu_delta_t = delta_t * mu;
    mu_2 = -2 * mu;

    /* derivative of bound */
    for (j = 1; j < k_max; ++j) {
        bound_deriv[j - 1] = (bound[j] - bound[j - 1]) / delta_t;
    }
    bound_deriv[k_max - 1] = bound_deriv[k_max - 2];

    /* norm_sqrt_t[i] = 1 / sqrt(2 * pi * delta_t * (i + 1)) 
       norm_t[i] = 1 / (delta_t * (i + 1)) */
    for (j = 0; j < k_max; ++j) {
        norm_sqrt_t[j] = 1.0 / sqrt(pi_delta_t_2 * (j + 1.0));
        norm_t[j] = 1.0 / (delta_t * (j + 1.0));
    }

    /* fill g1 recursively, g2 is based on g1 */
    for (k = 0; k < k_max; ++k) {
        /* speed increase by reducing array access */
        double bound_k = bound[k];
        double bound_deriv_k1 = bound_deriv[k] - mu;
        double bound_deriv_k2 = -bound_deriv[k] - mu;
        double cum_mu_k = (k + 1) * mu_delta_t;
        double norm_t_j = norm_t[k];
        double norm_sqrt_t_j = norm_sqrt_t[k];
        /* initial values */
        double g1_k = -norm_sqrt_t_j
                      * exp(- 0.5 * (bound_k - cum_mu_k) * (bound_k - cum_mu_k) * norm_t_j)
                      * (bound_deriv_k1 - (bound_k - cum_mu_k) * norm_t_j);
        /* relation to previous values */
        for (j = 0; j < k; ++j) {
            /* reducing array access + pre-compute values */
            double bound_j = bound[j];
            double cum_mu_k_j = (k - j) * mu_delta_t;
            double diff1 = bound_k - bound_j - cum_mu_k_j;
            double diff2 = bound_k + bound_j - cum_mu_k_j;
            norm_t_j = norm_t[k - j - 1];
            norm_sqrt_t_j = norm_sqrt_t[k - j - 1];
            /* add values */
            g1_k += delta_t * norm_sqrt_t_j
                    * (g1[j] * exp(- 0.5 * diff1 * diff1 * norm_t_j)
                       * (bound_deriv_k1 - diff1 * norm_t_j)
                       + g2[j] * exp(- 0.5 * diff2 * diff2 * norm_t_j)
                       * (bound_deriv_k1 - diff2 * norm_t_j));
        }
        /* avoid negative densities that could appear due to numerical instab. */
        g1[k] = MAX(g1_k, 0);
        g2[k] = MAX(g1_k * exp(mu_2 * bound_k), 0);
    }
    
    free(bound_deriv);
    free(norm_sqrt_t);
    free(norm_t);
    return 0;
}


/** ddm_rt_dist_const - reaction time distribution for constant drift/bound
 * mu - drift rate
 * bound - bound (duh)
 * delta_t - step size in [s]
 * k_max - number of steps, t_max = k_max * delta_t
 * Results are stored in
 * g1 - vector (size k_max) for pdf of first passage times, upper boundary
 * g2 - vector (size k_max) for pdf of first passage times, lower boundary
 */
void ddm_rt_dist_const(double mu, double bound, double delta_t, int k_max,
             double g1[], double g2[])
{
    double mu_bound, exp_mu_bound, sqrt_2_pi_bound, t;
    int i;
    
    assert(mu > 0 && bound > 0 && delta_t > 0 && k_max > 0 &&
    g1 != NULL && g2 != NULL);
    
    /* some constants */
    mu_bound = -2 * mu * bound;
    exp_mu_bound = exp(mu_bound);
    sqrt_2_pi_bound = bound / sqrt(PI_2);
    
    /* series expansion approximation from Cox & Miller (1977) */    
    t = delta_t;
    for (i = 0; i < k_max; ++i) {
        double g = 0;
        int j = 0;
        double incr = SERIES_ACC + 1.0;
        /* iterate until certain accuracy is reached */
        while (incr >= SERIES_ACC) {
            double x = ((2 * j + 1) * bound - mu * t);
            incr = (2 * j + 1) * exp(mu_bound * j) * exp(- 0.5 / t * x * x);
            g += ((j % 2) == 0 ? incr : (-incr));
            j++;
        }
        g *= sqrt_2_pi_bound * pow(t, -1.5);
        (*g1++) = MAX(g, 0);
        (*g2++) = MAX(exp_mu_bound * g, 0);
        t += delta_t;
    }
}


/** ddm_rt_dist - compute the reaction time distribution, weighted input
 * mu - vector of drift rates, of size k_max, in steps of delta_t
 * bound - vector of bound heights, of size k_max, in steps of delta_t
 * k - proportionality factor
 * delta_t - step size in [s]
 * n_max - number of steps, t_max = n_max * delta_t
 * Results are stored in
 * g1 - vector (size n_max) for pdf of first passage times, upper boundary
 * g2 - vector (size n_max) for pdf of first passage times, lower boundary
 *
 * 0 is returned on success, -1 on memory allocation failure
 */
int ddm_rt_dist_w(double mu[], double bound[], double k, double delta_t,
                  int n_max, double g1[], double g2[])
{
    double k_2, cum_a2;
    double *a2, *A, *bound_deriv;
    int j, n;
    
    assert(mu != NULL && bound != NULL && delta_t > 0.0 &&
           n_max > 0 &&  g1 != NULL && g2 != NULL);
    
    /* allocate array memory */
    a2 = malloc(n_max * sizeof(double));
    A = malloc(n_max * sizeof(double));
    bound_deriv = malloc(n_max * sizeof(double));
    if (a2 == NULL || A == NULL || bound_deriv == NULL) {
        free(a2);
        free(A);
        free(bound_deriv);
        return -1;
    }

    /* pre-compute value */
    k_2 = -2 * k;

    /* a2(t) = mu(t)^2, A_t(t) = \int^t a2(s) ds, and derivative of bound */
    cum_a2 = delta_t * (a2[0] = mu[0] * mu[0]);
    A[0] = cum_a2;
    for (j = 1; j < n_max; ++j) {
        cum_a2 += delta_t * (a2[j] = mu[j] * mu[j]);
        A[j] = cum_a2;
        bound_deriv[j - 1] = (bound[j] - bound[j - 1]) / delta_t;
    }
    bound_deriv[n_max - 1] = bound_deriv[n_max - 2];

    /* fill up g1 and g2 recursively */
    for (n = 0; n < n_max; ++n) {
        /* reduce array access */
        double bound_n = bound[n];
        double a2_n = a2[n];
        double A_n = A[n];
        double bound_deriv_n = bound_deriv[n];

        /* initial values */
        double diff1 = bound_n - k * A_n;
        double diff2 = -bound_n - k * A_n;
        double sqrt_A_n = sqrt(PI_2 * A_n);
        double tmp = bound_deriv_n - bound_n / A_n * a2_n;
        double g1_n = - exp(-0.5 * diff1 * diff1 / A_n) / sqrt_A_n * tmp;

        /* relation to previous values */
        for (j = 0; j < n; ++j) {
            /* reduce array access and pre-compute values */
            double diff1_A, diff2_A;
            double bound_j = bound[j];
            double A_diff = A_n - A[j];
            double sqrt_A_diff = sqrt(PI_2 * A_diff);
            diff1 = bound_n - bound_j;
            diff2 = bound_n + bound_j;

            diff1_A = diff1 - k * A_diff;
            diff2_A = diff2 - k * A_diff;
            g1_n += delta_t / sqrt_A_diff * (
                        g1[j] * exp(-0.5 * diff1_A * diff1_A / A_diff) 
                        * (bound_deriv_n - a2_n * diff1 / A_diff)
                      + g2[j] * exp(-0.5 * diff2_A * diff2_A / A_diff)
                        * (bound_deriv_n - a2_n * diff2 / A_diff));
        }
        /* avoid negative densities that could appear due to numerical instab. */
        g1[n] = MAX(g1_n, 0);
        g2[n] = MAX(g1_n * exp(k_2 * bound_n), 0);
    }

    free(a2);
    free(A);
    free(bound_deriv);
    return 0;
}

