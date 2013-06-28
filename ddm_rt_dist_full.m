[g1, g2] = ddm_rt_dist_full(mu, sig2, b_lo, b_up, b_lo_deriv, b_up_deriv, ...
                            delta_t, t_max, inv_leak)
%% computes the diffusion model first-passage time distributions
%
% The applied method is described in Smith (2000) "Stochastic Dynamic
% Models of Response Time and Accuracy: A Foundational Primer" and other
% sources. Drift rate, bounds, and diffusion variance are allowed to vary
% over time.
%
% [g1, g2] = ddm_rt_dist_full(mu, sig2, b_lo, b_up, b_lo_deriv, b_up_deriv,
%                             delta_t, t_max, [inv_leak])
%
% mu, ..., b_up_deriv are all vectors in steps of delta_t. mu and sig2 are
% the drift rate and variance, respectively. b_lo and b_up are the location
% of the lower and upper bound, and b_lo_deriv and b_up_deriv are their
% time derivatives. t_max is the maximum time up until which the reaction
% time distributions are evaluated. g1 and g2 hold the probability
% densities of hitting the upper and lower bound, respectively, in steps of
% delta_t up to and including t_max. If the given vectors are shorter than
% t_max, their last element is replicated (except for b_lo_deriv /
% b_up_deriv, whose last element is set to 0).
%
% If inv_leak is given, a leaky integator rather than a non-leaky one is
% assumed. In this case, inv_leak is 1 / leak time constant. The non-leaky
% case is the same as inv_leak = 0, but uses a different algorithm to
% compute the probability densities.
%
% The assumed model is
%
% dx / dt = - inv_leak * x(t) + mu(t) + sqrt(sig2(t)) eta(t)
%
% where eta is zero-mean unit variance white noise. The bound is on x.
% inv_leak defaults to 0 if not given.
%
%
% [g1, g2] = ddm_rt_dist(..., 'mnorm', 'yes')
%
% Causes both g1 and g2 to be normalised such that the densities integrate
% to 1. The normalisation is performed by adding all missing mass to the
% last element of g1 / g2, such that the proportion of the mass in g1 and
% g2 remains unchanged. This is useful if there is some significant portion
% of the mass expected to occur after t_max. By default, 'mnorm' is set to
% 'no'.

error('Not implemented as M-file. Make sure that mex file is complied');