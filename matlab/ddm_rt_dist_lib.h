/**
 * Copyright (c) 2013, Jan Drugowitsch
 * All rights reserved.
 * See the file LICENSE for licensing information.
 *   
 * ddm_rt_dist_lib.h
 *
 *
 * 2012-02-28 Jan Drugowitsch   initial release v0.1
 **/

#ifndef _DDM_RT_DIST_LIB_H_
#define _DDM_RT_DIST_LIB_H_

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
                     double delta_t, int k_max, double g1[], double g2[]);

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
                          double g1[], double g2[]);

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
             double g1[], double g2[]);

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
             double g1[], double g2[]);

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
             double g1[], double g2[]);

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
                  int n_max, double g1[], double g2[]);

#endif
