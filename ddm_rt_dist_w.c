/**
 * Copyright (c) 2013, Jan Drugowitsch
 * All rights reserved.
 * See the file LICENSE for licensing information.
 *   
 * ddm_rt_dist_w.c - comuting the DDM reaction time distribution as described
 *                   in Smith (2000) "Stochastic Dynamic Models of Response
 *                   Time and  Accurary: A Foundational Primer". The assumed
 *                   diffusion model is not standard and is described below.
 *
 * [g1, g2] = ddm_rt_dist_w(a, bound, k, delta_t, t_max)
 *
 * a and bound are vectors of evidence strength and bound height over time,
 * in steps of delta_t. k is the proportionality factor that turns the evidence
 * strength into a drift rate. t_max is the maximum time up until which the
 * first passage time distributions are evaluated. g1 and g2 hold these
 * distributions, for hitting the upper and lower bound, respectively, in
 * steps of delta_t up to and including t_max. The the vectors mu and bound are
 * shorter than t_max, their last elements are replicated.
 *
 * The underlying model is given by
 *
 * dx / dt = a(t) * (k a(t) + eta(t)),
 *
 * where eta(t) is a white noise process. The bounds on x are located at bound(t)
 * and -bound(t).
 */

#include "mex.h"
#include "matrix.h"

#include "ddm_rt_dist_lib.h"

#include <math.h>
#include <string.h>
#include <assert.h>

#define MEX_ARGIN_IS_REAL_DOUBLE(arg_idx) (mxIsDouble(prhs[arg_idx]) && !mxIsComplex(prhs[arg_idx]) && mxGetN(prhs[arg_idx]) == 1 && mxGetM(prhs[arg_idx]) == 1)
#define MEX_ARGIN_IS_REAL_VECTOR(arg_idx) (mxIsDouble(prhs[arg_idx]) && !mxIsComplex(prhs[arg_idx]) && ((mxGetN(prhs[arg_idx]) == 1 && mxGetM(prhs[arg_idx]) >= 1) || (mxGetN(prhs[arg_idx]) >= 1 && mxGetM(prhs[arg_idx]) == 1)))

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


/* the gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mexWarnMsgTxt("ddm_rt_dist_w() is depricated. Use ddm_rt_dist() instead.");
    
    /* [g1, g2] = ddm_rt_dist_w(mu, bound, k, delta_t, t_max) */
    /* Check argument number */
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("ddm_rt_dist_w:WrongOutputs", 
                          "Wrong number of output arguments");
    }
    if (nrhs != 5) {
        mexErrMsgIdAndTxt("ddm_rt_dist_w:WrongInputs",
                          "Wrong number of input arguments");
    }

    /* Check individual arguments */
    if (!MEX_ARGIN_IS_REAL_VECTOR(0))
        mexErrMsgIdAndTxt("ddm_rt_dist_w:WrongInput",
                          "First input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_VECTOR(1))
        mexErrMsgIdAndTxt("ddm_rt_dist_w:WrongInput",
                          "Second input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_DOUBLE(2))
        mexErrMsgIdAndTxt("ddm_rt_dist_w:WrongInput",
                          "Third input argument expected to be a double");
    if (!MEX_ARGIN_IS_REAL_DOUBLE(3))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Fourth input argument expected to be a double");
    if (!MEX_ARGIN_IS_REAL_DOUBLE(4))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Fifth input argument expected to be a double");

    /* Process arguments */
    int mu_size = MAX(mxGetN(prhs[0]), mxGetM(prhs[0]));
    double *mu = mxGetPr(prhs[0]);
    int bound_size = MAX(mxGetN(prhs[1]), mxGetM(prhs[1]));
    double *bound = mxGetPr(prhs[1]);
    double k = mxGetScalar(prhs[2]);
    double delta_t = mxGetScalar(prhs[3]);
    double t_max = mxGetScalar(prhs[4]);
    if (delta_t <= 0.0)
        mexErrMsgIdAndTxt("ddm_rt_dist_w:WrongInput",
                          "delta_t needs to be larger than 0.0");
    if (t_max <= delta_t)
        mexErrMsgIdAndTxt("ddm_rt_dist_w:WrongInput",
                          "t_max needs to be at least as large as delta_t");

    /* extend mu and bound by replicating last element, if necessary */
    int n_max = (int) ceil(t_max / delta_t);

    double mu_ext[n_max];
    memcpy(mu_ext, mu, sizeof(double) * MIN(mu_size, n_max));
    int i;
    double last_mu = mu[mu_size - 1];
    for (i = mu_size; i < n_max; ++i)
        mu_ext[i] = last_mu;

    double bound_ext[n_max];
    memcpy(bound_ext, bound, sizeof(double) * MIN(bound_size, n_max));
    double last_bound = bound[bound_size - 1];
    for (i = bound_size; i < n_max; ++i)
        bound_ext[i] = last_bound;

    /* reserve space for output */
    plhs[0] = mxCreateDoubleMatrix(1, n_max, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, n_max, mxREAL);

    /* compute the pdf's */
    ddm_rt_dist_w(mu_ext, bound_ext, k, delta_t, n_max, 
                  mxGetPr(plhs[0]), mxGetPr(plhs[1]));
}
