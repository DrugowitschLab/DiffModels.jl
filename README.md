DiffModels.jl
=============

[![Build Status](https://travis-ci.org/DrugowitschLab/DiffModels.jl.svg?branch=master)](https://travis-ci.org/DrugowitschLab/DiffModels.jl)

[![Coverage Status](https://coveralls.io/repos/DrugowitschLab/DiffModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/DrugowitschLab/DiffModels.jl?branch=master)

[![codecov.io](http://codecov.io/github/DrugowitschLab/DiffModels.jl/coverage.svg?branch=master)](http://codecov.io/github/DrugowitschLab/DiffModels.jl?branch=master)

A Julia package for simulating diffusion models and compute their first passage time densities.

*No guarantee is provided for the correctness of the implementation.*

The code is licensed under the MIT License.

Content
-------

The library provides Julia classes and functions to define diffusion models, sample their first-passage times and boundaries, and compute their first-passage time density functions. For now, only diffusion models with two absorbing boundaries are supported. The library supports time-changing boundaries and drifts.

The diffusion models assume a drifting and diffusing particle *x(t)* that starts at *x(0) = 0*and whose time-course follows

*dx = mu(t) dt + sig(t) dW* ,

where *mu(t)* is the current drift, *sig(t)* is the current diffusion standard deviation (for now assumed to be *sig(t)=1*, always), and *dW* is a Wiener process. Diffusion is terminated as soon as the particle reaches either the upper boundary *theta_u(t)* or lower boundary *theta_t(t)*. The library requires *theta_u(0) > 0* and *theta_l(0) < 0*. The time at which either boundary is reached is the first-passage time. The associated densities, *g_u(t)* and *g_l(t)*, are the joint densities over bounds and first-passage times, such that

*integral_0^infinity (g_u(t) + g_l(t)) dt = 1* .

The library provides specialised classes for time-invariant drifts, *mu(t) = mu_0* for all *t*, time-invariant bounds, *theta(t) = theta_0* for all *t*, symmetric bounds, *theta_l(t) = - theta_u(t)* (see below).

Installation
------------

The easiest way to install DiffModels.jl is by using the Julia Package Manager at the Julia Pkg prompt:
```
pkg> add https://github.com/DrugowitschLab/DiffModels.jl
```

Usage
-----

The library is based on assembling diffusion models from drift and boundary specifications. If not constant, all drifts/boundaries are specified by vectors in time steps of *dt*, with the first vector element corresponding to *t=0*. In most cases, these vectors need to be sufficiently long to cover the whole time of relevance. The first-passage times cannot computed beyond this time, and samples cannot be drawn after it. This restriction does not apply to time-invariant drifts/bounds, which do not feature this limitation.

### Drift

The available drifts are defined in [src/drift.jl](src/drift.jl). All drifts are based on the abstract base class `AbstractDrift`. The following constructors are available:
```Julia
ConstDrift(mu::Real, dt::Real)
VarDrift(mu::Vector{Float64}, dt::Real)
VarDrift(mu::Vector{Float64}, dt::Real, maxt::Real)
```
`ConstDrift` is a constant drift of size `mu`. `dt` needs to be nonetheless specified, to evaluate the step size for diffusion model sampling. `VarDrift` is a drift that changes over time according to the `mu` vector that specifies this drift in steps of `dt`. If `maxt` is given, the `mu` vector is extended by repeating its last element until time `maxt`.

### Boundaries

The available boundaries are defined in [src/bound.jl](src/bound.jl). Single boundaries are based on the abstract base class 'AbstractBound'. For such boundaries, the following constructors are available:
```Julia
ConstBound(b::Real, dt::Real)
LinearBound(b0::Real, bslope::Real, dt::Real)
VarBound(b::Vector{Float64}, bg::Vector{Float64}, dt::Real)
VarBound(b::Vector{Float64}, dt::Real)
```
`ConstBound` is a constant boundary at `b`. `LinearBound` is a linearly changing boundary that, at time `t` is located at `b0 + t * bslope`. `VarBound` is a time-varying boundary that changes over time according to the vector `b` in steps of `dt`. `bg` is its time derivative, and needs to contain the same number of elements as `b`. If not specified (last constructor), it is estimated from `b` by finite differences.

Boundary pairs are based on the abstract base class `AbstractBounds`. For such pairs, the following constructors are available:
```Julia
SymBounds(b::T) where T <: AbstractBound
typealias VarSymBounds SymBounds{VarBound}
typealias LinearSymBounds SymBounds{LinearBound}
typealias ConstSymBounds SymBounds{ConstBound}

AsymBounds(upper::T1, lower::T2) where {T1 <: AbstractBound, T2 <: AbstractBound}
VarAsymBounds AsymBounds{VarBound, VarBound}
ConstAsymBounds AsymBounds{ConstBound, ConstBound}

typealias ConstBounds Union(ConstSymBounds, ConstAsymBounds)
```
`SymBounds` and `AsymBounds` specify symmetric boundaries (around zero) and asymmetric boundaries, respectively. `SymBounds` needs to be constructed with the upper boundary, and the lower boundaries is mirrored around zero. `AsymBounds` is constructed with two boundaries, where `upper` is the upper boundary, and `lower` is the *negative* lower boundary.

### First-passage time densities

In the most general case, the first-passage time densities are computed by
```Julia
pdf(d::AbstractDrift, b::AbstractBounds, tmax::Real)
```
This function returns the vectors `g1` and `g2` that contain the densities for the upper and the lower boundary, respecively, in steps of `dt` (as specified by drift and bounds) up to time `tmax`.

For some combination of constant drifts and constant boundaries, significantly more efficient functions are available:
```Julia
pdf(d::ConstDrift, b::ConstBounds, tmax::Real)
pdfu(d::ConstDrift, b::ConstBounds, t::Float64)
pdfl(d::ConstDrift, b::ConstBounds, t::Float64)
pdful(d::ConstDrift, b::ConstBounds, t::Float64)
```
`pdf` returns two vectors, `g1` and `g2`, as before. `pdfu` and `pdfl` compute the first-passage time density at the upper and lower boundary, respecively, only at time `t`. `pdful` returns both densities for this time.

### Drawing first-passage time and boundary samples

Diffusion model sampling is based on the `Sampler` framework from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). The basic idea to is create a sampler `s` from a drift/boundary specification, on which `rand(s)` is called. All calls to `rand(s)` return a `t, bound` pair, where `t` is the first-passage time, and `bound` is `true` if the upper boundary was reached, and `false` otherwise. Currently, the following samplers exist:
```Julia
sampler(d::AbstractDrift, b::AbstractBounds)
sampler(d::ConstDrift, b::ConstSymBounds)
sampler(d::ConstDrift, b::ConstAsymBounds)
```
The first sampler is a generic diffusion model sampler that draws samples by simulating full trajectories in steps of `dt`. All samples beyond the time that the drift/bound are specified return `t = Inf` and a random `bound`. The other samplers use a specialised and significantly faster method, based on rejection sampling.

References
----------

In general, the library computes the first-passage time densities by finding the solution to an integral equation, as described in

Smith PL (2000). [Stochastic Dynamic Models of Response Time and Accuracy: A Foundational Primer](http://dx.doi.org/10.1006/jmps.1999.1260). *Journal of Mathematical Psychology*, 44 (3). 408-463.

For constant drift and bounds, it instead uses a much faster method, based on an infinite series expansion of these densities, as described in

Cox DR and Miller HD (1965). *The Theory of Stochastic Processes*. John Wiley & Sons, Inc.

and

Navarro DJ and Fuss IG (2009). [Fast and accurate calculations for first-passage times in Wiener diffusion models](http://dx.doi.org/10.1016/j.jmp.2009.02.003). *Journal of Mathematical Psychology*, 53, 222-230.

Samples are in the most general case drawn by simulating trajectories by the Euler–Maruyama method. For diffusion models with constant drift and (symmetric or asymmetric) boundaries, the following significantly faster method based on rejection sampling is used:

Drugowitsch J (2016). [Fast and accurate Monte Carlo sampling of first-passage times from Wiener diffusion models](http://dx.doi.org/10.1038/srep20490). *Scientific Reports* 6, 20490; doi: 10.1038/srep20490.
