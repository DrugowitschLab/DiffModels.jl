function [g1, g2] = ddm_rt_dist(mu, bound, delta_t, t_max)
%% [g1, g2] = ddm_rt_dist(mu, bound, delta_t, t_max, ...)
%
% comuting the diffusion model first-passage times as described in Smith
% (2000) "Stochastic Dynamic Models of Response Time and  Accurary: A
% Foundational Primer" and other sources. Both the drift rate and the
% (symmetric) bound can vary over time. A variant for weighted accumulation
% is also provided.
%
%
% [g1, g2] = ddm_rt_dist(mu, bound, delta_t, t_max, ...)
%
% mu and bound are vectors of drift rates and bound height over time, in
% steps of delta_t. t_max is the maximum time up until which the reaction time
% distributions are evaluated. g1 and g2 hold the probability densities of
% hitting the upper bound and lower bound, respectively, in steps of delta_t
% up to and including t_max. If the vectors mu and bound are shorter than
% t_max, their last elements replicated.
%
% The assumed model is
%
% dx / dt = mu(t) + eta(t)
%
% where eta is zero-mean unit variance white noise. The bound is on x and -x.
%
% The method uses more efficient methods of computing the reaction time
% density if either mu is constant (i.e. given as a scalar) or both mu and
% the bound are constant.
%
% 
% [g1, g2] = ddm_rt_dist(a, bound, delta_t, t_max, k, ...)
%
% Performs weighted accumulation with weights given by vector a. k is a scalar
% that determines the proportionality constant. The assumed model is
%
% dz / dt = k a(t) + eta(t)
% dx / dt = a(t) dz / dt
%
% The bound is on x and -x.
%
%
% [g1, g2] = ddm_rt_dist(..., 'mnorm', 'yes')
%
% Causes both g1 and g2 to be normalised such that the densities integrate to
% 1. The normalisation is performed by adding all missing mass to the last
% element of g1 / g2, such that the proportion of the mass in g1 and g2
% remains unchanged. This is useful if there is some significant portion of
% the mass expected to occur after t_max. By default, 'mnorm' is set to 'no'.
%
% Copyright (c) 2013, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.

error('Not implemented as M-file. Make sure that mex file is complied');
