/**
 * ddm_rt_dist_full.c - comuting the DDM reaction time distribution as described
 *                 in Smith (2000) "Stochastic Dynamic Models of Response Time
 *                 and Accurary: A Foundational Primer" and other sources. Drift
 *                 rate, bounds, and diffusion variance is allowed to vary over
 *                 time.
 *
 * [g1, g2] = ddm_rt_dist_full(mu, sig2, b_lo, b_up, b_lo_deriv, b_up_deriv,
 *                             delta_t, t_max, [inv_leak])
 *
 * mu, ..., b_up_deriv are all vectors in steps of delta_t. mu and sig2 are the
 * drift rate and variance, respectively. b_lo and b_up are the location of the
 * lower and upper bound, and b_lo_deriv and b_up_deriv are their time
 * derivatives. t_max is the maximum time up until which the reaction time
 * distributions are evaluated. g1 and g2 hold the probability densities of
 * hitting the upper and lower bound, respectively, in steps of delta_t up to
 * and including t_max. If the given vectors are shorter than t_max, their last
 * element is replicated (except for b_lo_deriv / b_up_deriv, whose last element
 * is set to 0).
 * 
 * If inv_leak is given, a leaky integator rather than a non-leaky one is
 * assumed. In this case, inv_leak is 1 / leak time constant. The non-leaky
 * case is the same as inv_leak = 0, but uses a different algorithm to compute
 * the probability densities.
 *
 * The assumed model is
 *
 * dx / dt = - inv_leak * x(t) + mu(t) + sqrt(sig2(t)) eta(t)
 *
 * where eta is zero-mean unit variance white noise. The bound is on x.
 * inv_leak defaults to 0 if not given.
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


/** creates a new vector, copies v, and fills the rest with fill_el
 * 
 * The new vector is of size new_size. If v_size > new_size then not all
 * elements of v are copied. If v_size < new_size, then the elements of the
 * new vector are filled up with fill_el.
 * 
 * The function returns NULL if it fails to allocate memory for the new vector.
 **/
double* extend_vector(double v[], int v_size, int new_size, double fill_el)
{
    double *new_v;
    int i;
    
    new_v = malloc(new_size * sizeof(double));
    if (new_v == NULL)
        return NULL;
    
    memcpy(new_v, v, sizeof(double) * MIN(v_size, new_size));
    for (i = v_size; i < new_size; ++i)
        new_v[i] = fill_el;
    
    return new_v;
}


/** the gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int mu_size, sig2_size, k_max, err, cur_argin, normalise_mass = 0;
    int b_lo_size, b_up_size, b_lo_deriv_size, b_up_deriv_size, has_leak = 0;
    double *mu, *sig2, *b_lo, *b_up, *b_lo_deriv, *b_up_deriv, *mu_ext;
    double *sig2_ext, *b_lo_ext, *b_up_ext, *b_lo_deriv_ext, *b_up_deriv_ext;
    double delta_t, t_max, inv_leak;
    /* [g1, g2] = rt_dist_full(mu, sig2, b_lo, b_up, b_lo_deriv, b_up_deriv,
                              delta_t, t_max, [leak]) */

    /* Check argument number */
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongOutputs", 
                          "Wrong number of output arguments");
    }
    if (nrhs < 8) {
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInputs",
                          "Too few input arguments");
    }

    /* Process first 8 arguments */
    if (!MEX_ARGIN_IS_REAL_VECTOR(0))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "First input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_VECTOR(1))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Second input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_VECTOR(2))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Third input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_VECTOR(3))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Fourth input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_VECTOR(4))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Fifth input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_VECTOR(5))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Sixth input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_DOUBLE(6))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Seventh input argument expected to be a double");
    if (!MEX_ARGIN_IS_REAL_DOUBLE(7))
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "Eight input argument expected to be a double");
    mu_size = MAX(mxGetN(prhs[0]), mxGetM(prhs[0]));
    sig2_size = MAX(mxGetN(prhs[1]), mxGetM(prhs[1]));
    b_lo_size = MAX(mxGetN(prhs[2]), mxGetM(prhs[2]));
    b_up_size = MAX(mxGetN(prhs[3]), mxGetM(prhs[3]));
    b_lo_deriv_size = MAX(mxGetN(prhs[4]), mxGetM(prhs[4]));
    b_up_deriv_size = MAX(mxGetN(prhs[5]), mxGetM(prhs[5]));
    mu = mxGetPr(prhs[0]);
    sig2 = mxGetPr(prhs[1]);
    b_lo = mxGetPr(prhs[2]);
    b_up = mxGetPr(prhs[3]);
    b_lo_deriv = mxGetPr(prhs[4]);
    b_up_deriv = mxGetPr(prhs[5]);
    delta_t = mxGetScalar(prhs[6]);
    t_max = mxGetScalar(prhs[7]);
    if (delta_t <= 0.0)
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "delta_t needs to be larger than 0.0");
    if (t_max <= delta_t)
        mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                          "t_max needs to be at least as large as delta_t");

    /* Process possible 9th non-string argument */
    cur_argin = 8;
    if (nrhs > cur_argin && !mxIsChar(prhs[cur_argin])) {
        if (!MEX_ARGIN_IS_REAL_DOUBLE(cur_argin))
            mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                              "Ninth input argument expected to be a double");
        inv_leak = mxGetScalar(prhs[cur_argin]);
        if (inv_leak < 0.0)
            mexErrMsgIdAndTxt("ddm_rt_dist:WrongInput",
                              "inv_leak needs to be non-negative");
        has_leak = 1;
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
    
    /* extend vectors by replicating last element */
    mu_ext = extend_vector(mu, mu_size, k_max, mu[mu_size - 1]);
    sig2_ext = extend_vector(sig2, sig2_size, k_max, sig2[sig2_size - 1]);
    b_lo_ext = extend_vector(b_lo, b_lo_size, k_max, b_lo[b_lo_size - 1]);
    b_up_ext = extend_vector(b_up, b_up_size, k_max, b_up[b_up_size - 1]);
    b_lo_deriv_ext = extend_vector(b_lo_deriv, b_lo_deriv_size, k_max, 0.0);
    b_up_deriv_ext = extend_vector(b_up_deriv, b_up_deriv_size, k_max, 0.0);
    if (mu_ext == NULL || sig2_ext == NULL ||
        b_lo_ext == NULL || b_up_ext == NULL ||
        b_lo_deriv_ext == NULL || b_up_deriv_ext == NULL) {
        free(mu_ext);
        free(sig2_ext);
        free(b_lo_ext);
        free(b_up_ext);
        free(b_lo_deriv_ext);
        free(b_up_deriv_ext);
        mexErrMsgIdAndTxt("ddm_rt_dist:OutOfMemory", "Out of memory");
    }
    
    /* compute the pdf's */
    if (has_leak)
        err = ddm_rt_dist_full_leak(mu_ext, sig2_ext, b_lo_ext, b_up_ext,
                                    b_lo_deriv_ext, b_up_deriv_ext,
                                    inv_leak, delta_t, k_max,
                                    mxGetPr(plhs[0]), mxGetPr(plhs[1]));
    else
        err = ddm_rt_dist_full(mu_ext, sig2_ext, b_lo_ext, b_up_ext,
                               b_lo_deriv_ext, b_up_deriv_ext, delta_t, k_max,
                               mxGetPr(plhs[0]), mxGetPr(plhs[1]));
    
    free(mu_ext);
    free(sig2_ext);
    free(b_lo_ext);
    free(b_up_ext);
    free(b_lo_deriv_ext);
    free(b_up_deriv_ext);

    if (err == -1)
        mexErrMsgIdAndTxt("ddm_rt_dist:OutOfMemory", "Out of memory");

    
    /* normalise mass, if requested */
    if (normalise_mass)
        mnorm(mxGetPr(plhs[0]), mxGetPr(plhs[1]), k_max, delta_t);
}
