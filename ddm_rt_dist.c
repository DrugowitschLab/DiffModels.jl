/**
 * Copyright (c) 2013, Jan Drugowitsch
 * All rights reserved.
 * See the file LICENSE for licensing information.
 *
 * ddm_rt_dist.c - comuting the DDM reaction time distribution as described in
 *                 Smith (2000) "Stochastic Dynamic Models of Response Time and 
 *                 Accurary: A Foundational Primer" and other sources. Both the
 *                 drift rate and the (symmetric) bound can vary over time.
 *                 A variant for weighted accumulation is also provided.
 *
 * [g1, g2] = ddm_rt_dist(mu, bound, delta_t, t_max, ...)
 *
 * mu and bound are vectors of drift rates and bound height over time, in steps
 * of delta_t. t_max is the maximum time up until which the reaction time
 * distributions are evaluated. g1 and g2 hold the probability densities of
 * hitting the upper bound and lower bound, respectively, in steps of delta_t
 * up to and including t_max. If the vectors mu and bound are shorter than
 * t_max, their last elements replicated.
 *
 * The assumed model is
 *
 * dx / dt = mu(t) + eta(t)
 *
 * where eta is zero-mean unit variance white noise. The bound is on x and -x.
 *
 * The method uses more efficient methods of computing the reaction time
 * density if either mu is constant (i.e. given as a scalar) or both mu and
 * the bound are constant.
 *
 *
 * [g1, g2] = ddm_rt_dist(a, bound, delta_t, t_max, k, ...)
 *
 * Performs weighted accumulation with weights given by vector a. k is a scalar
 * that determines the proportionality constant. The assumed model is
 *
 * dz / dt = k a(t) + eta(t)
 * dx / dt = a(t) dz / dt
 *
 * The bound is on x and -x.
 *
 * [g1, g2] = ddm_rt_dist(..., 'mnorm', 'yes')
 *
 * Causes both g1 and g2 to be normalised such that the densities integrate to
 * 1. The normalisation is performed by adding all missing mass to the last
 * element of g1 / g2, such that the proportion of the mass in g1 and g2
 * remains unchanged. This is useful if there is some significant portion of
 * the mass expected to occur after t_max. By default, 'mnorm' is set to 'no'. 
 *
 * 2012-02-28 Jan Drugowitsch   initial release v0.1
 */

#include "mex.h"
#include "matrix.h"

#include "ddm_rt_dist_lib.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define MEX_ARGIN_IS_REAL_DOUBLE(arg_idx) (mxIsDouble(prhs[arg_idx]) && !mxIsComplex(prhs[arg_idx]) && mxGetN(prhs[arg_idx]) == 1 && mxGetM(prhs[arg_idx]) == 1)
#define MEX_ARGIN_IS_REAL_VECTOR(arg_idx) (mxIsDouble(prhs[arg_idx]) && !mxIsComplex(prhs[arg_idx]) && ((mxGetN(prhs[arg_idx]) == 1 && mxGetM(prhs[arg_idx]) >= 1) || (mxGetN(prhs[arg_idx]) >= 1 && mxGetM(prhs[arg_idx]) == 1)))

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


/** normalising the mass, such that (sum(g1) + sum(g2) * delta_t = 1 
 *
 * Function makes sure that g1(t) >= 0, g2(t) >= 0, for all t, and that
 * (sum(g1) + sum(g2) * delta_t) = 1. It does so by eventually adding mass to
 * the last elements of g1 / g2, such that the ratio
 * sum(g1) / (sum(g1) + sum(g2)) (after removing negative values) remains
 * unchanged.
 */
void mnorm(double g1[], double g2[], int n, double delta_t)
{
    /* remove negative elements and compute sum */
    double g1_sum = 0.0, g2_sum = 0.0, p;
    int i;
    for (i = 0; i < n; ++i) {
        if (g1[i] < 0) g1[i] = 0;
        else g1_sum += g1[i];
        if (g2[i] < 0) g2[i] = 0;
        else g2_sum += g2[i];
    }
    
    /* adjust last elements accoring to ratio */
    p = g1_sum / (g1_sum + g2_sum);
    g1[n - 1] += p / delta_t - g1_sum;
    g2[n - 1] += (1 - p) / delta_t - g2_sum;
}


/** the gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int mu_size, bound_size, cur_argin, k_max;
    int weighted_ddm = 0, normalise_mass = 0;
    double *mu, *bound, delta_t, t_max, k = 0.0;
    /* [g1, g2] = rt_dist(mu, bound, delta_t, t_max) or
       [g1, g2] = rt_dist(a, k, bound, delta_t, t_max) */

    /* Check argument number */
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongOutputs", 
                          "Wrong number of output arguments");
    }
    if (nrhs < 4) {
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInputs",
                          "Too few input arguments");
    }

    /* Process first 4 arguments */
    if (!MEX_ARGIN_IS_REAL_VECTOR(0))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "First input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_VECTOR(1))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Second input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_DOUBLE(2))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Third input argument expected to be a double");
    if (!MEX_ARGIN_IS_REAL_DOUBLE(3))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Forth input argument expected to be a double");
    mu_size = MAX(mxGetN(prhs[0]), mxGetM(prhs[0]));
    mu = mxGetPr(prhs[0]);
    bound_size = MAX(mxGetN(prhs[1]), mxGetM(prhs[1]));
    bound = mxGetPr(prhs[1]);
    delta_t = mxGetScalar(prhs[2]);
    t_max = mxGetScalar(prhs[3]);
    if (delta_t <= 0.0)
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "delta_t needs to be larger than 0.0");
    if (t_max <= delta_t)
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "t_max needs to be at least as large as delta_t");
    
    /* Process possible 5th non-string argument */
    cur_argin = 4;
    if (nrhs > 4 && !mxIsChar(prhs[4])) {
        if (!MEX_ARGIN_IS_REAL_DOUBLE(4))
            mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                              "Fifth input argument expected to be a double");
        k = mxGetScalar(prhs[4]);
        weighted_ddm = 1;
        ++cur_argin;
    }
    
    /* Process string arguments */
    if (nrhs > cur_argin) {
        char str_arg[6];
        /* current only accept 'mnorm' string argument */
        if (!mxIsChar(prhs[cur_argin]))
            mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                              "String argument expected but not found");
        if (mxGetString(prhs[cur_argin], str_arg, sizeof(str_arg)) == 1 ||
            strcmp(str_arg, "mnorm") != 0)
            mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                              "\"mnorm\" string argument expected");
        /* this needs to be followed by "yes" or "no" */
        if (nrhs <= cur_argin + 1 || !mxIsChar(prhs[cur_argin + 1]))
            mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                              "String expected after \"mnorm\"");
        if (mxGetString(prhs[cur_argin + 1], str_arg, sizeof(str_arg)) == 1 ||
            strcmp(str_arg, "yes") != 0 && strcmp(str_arg, "no") != 0)
            mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                              "\"yes\" or \"no\" expected after \"mnorm\"");
        normalise_mass = strcmp(str_arg, "yes") == 0;
        
        /* no arguments allowed after that */
        if (nrhs > cur_argin + 2)
            mexErrMsgIdAndTxt("ddm_rt_dist:WrongInputs",
                              "Too many input arguments");
    }

    /* extend mu and bound by replicating last element, if necessary */
    k_max = (int) ceil(t_max / delta_t);

    /* reserve space for output */
    plhs[0] = mxCreateDoubleMatrix(1, k_max, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, k_max, mxREAL);
    
    if (weighted_ddm) {
        /* extend mu and bound by replicating last element */
        double *mu_ext, *bound_ext, last_mu, last_bound;
        int i, err;
        
        mu_ext = malloc(k_max * sizeof(double));
        bound_ext = malloc(k_max * sizeof(double));
        if (mu_ext == NULL || bound_ext ==  NULL) {
            free(mu_ext);
            free(bound_ext);
            mexErrMsgIdAndTxt("ddm_rt_dist:OutOfMemory", "Out of memory");
        }

        memcpy(mu_ext, mu, sizeof(double) * MIN(mu_size, k_max));
        last_mu = mu[mu_size - 1];
        for (i = mu_size; i < k_max; ++i)
            mu_ext[i] = last_mu;

        memcpy(bound_ext, bound, sizeof(double) * MIN(bound_size, k_max));
        last_bound = bound[bound_size - 1];
        for (i = bound_size; i < k_max; ++i)
            bound_ext[i] = last_bound;

        /* compute the pdf's with weighted evidence */
        err = ddm_rt_dist_w(mu_ext, bound_ext, k, delta_t, k_max, 
                            mxGetPr(plhs[0]), mxGetPr(plhs[1]));
        
        free(mu_ext);
        free(bound_ext);
        
        if (err == -1)
            mexErrMsgIdAndTxt("ddm_rt_dist:OutOfMemory", "Out of memory");

    } else if (mu_size == 1) {
        if (bound_size == 1) {
            /* constant bound and drift - can use simpler method */
            ddm_rt_dist_const(mu[0], bound[0], delta_t, k_max,
                              mxGetPr(plhs[0]), mxGetPr(plhs[1]));
        } else {
            /* extend bound by replicating last element */
            double *bound_ext, last_bound;
            int i, err;
            
            bound_ext = malloc(k_max * sizeof(double));
            if (bound_ext == NULL)
                mexErrMsgIdAndTxt("ddm_rt_dist:OutOfMemory", "Out of memory");

            memcpy(bound_ext, bound, sizeof(double) * MIN(bound_size, k_max));
            last_bound = bound[bound_size - 1];
            for (i = bound_size; i < k_max; ++i)
                bound_ext[i] = last_bound;
            
            /* constant drift - slightly more efficient */
            err = ddm_rt_dist_const_mu(mu[0], bound_ext, delta_t, k_max,
                                       mxGetPr(plhs[0]), mxGetPr(plhs[1]));
            
            free(bound_ext);
            
            if (err == -1)
                mexErrMsgIdAndTxt("ddm_rt_dist:OutOfMemory", "Out of memory");
            
        }
    } else {
        /* extend mu and bound by replicating last element */
        double *mu_ext, *bound_ext, last_mu, last_bound;
        int i, err;
        
        mu_ext = malloc(k_max * sizeof(double));
        bound_ext = malloc(k_max * sizeof(double));
        if (mu_ext == NULL || bound_ext ==  NULL) {
            free(mu_ext);
            free(bound_ext);
            mexErrMsgIdAndTxt("ddm_rt_dist:OutOfMemory", "Out of memory");
        }

        memcpy(mu_ext, mu, sizeof(double) * MIN(mu_size, k_max));
        last_mu = mu[mu_size - 1];
        for (i = mu_size; i < k_max; ++i)
            mu_ext[i] = last_mu;

        memcpy(bound_ext, bound, sizeof(double) * MIN(bound_size, k_max));
        last_bound = bound[bound_size - 1];
        for (i = bound_size; i < k_max; ++i)
            bound_ext[i] = last_bound;

        /* compute the pdf's */
        err = ddm_rt_dist(mu_ext, bound_ext, delta_t, k_max, 
                          mxGetPr(plhs[0]), mxGetPr(plhs[1]));
        
        free(mu_ext);
        free(bound_ext);

        if (err == -1)
            mexErrMsgIdAndTxt("ddm_rt_dist:OutOfMemory", "Out of memory");
        
    }
    
    /* normalise mass, if requested */
    if (normalise_mass)
        mnorm(mxGetPr(plhs[0]), mxGetPr(plhs[1]), k_max, delta_t);
}
