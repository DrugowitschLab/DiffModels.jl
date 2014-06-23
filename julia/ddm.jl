# First-passage time distributions for diffusion models.

# TODO:
# - add version that includes leak
# - add version that includes time-varying variance
# - add bounds constructors with maxt, to extend/shorten given bounds
# - maybe define generic SymmetricBound class
# - add constant, non-symmetric bound, constant drift series expansion version
# - add simulation support, ev. using the Sampler framework

# constants

import Base.@math_const

@math_const twoπ   6.2831853071795864769 big(2.) * π
@math_const sqrt2π 2.5066282746310005024 sqrt(big(2.)*π)

# -----------------------------------------------------------------------------
# utility functions
# -----------------------------------------------------------------------------

# Returns the gradient of x, computed by finited differences
#
# Except for the first and last element, all gradients are computed by central
# finite differences
function fdgrad{T}(x::Vector{T}, dt::Real)
    const n = length(x)
    GType = typeof((zero(T) - zero(T)) / dt)
    g = Array(GType, n)
    if n == 1
        g[1] = zero(GType)
    else
        g[1] = (x[2] - x[1]) / dt
        for i = 2:(n-1)
            g[i] = (x[i+1] - x[i-1]) / 2dt
        end
        g[n] = (x[n] - x[n-1]) / dt
    end
    g
end


# -----------------------------------------------------------------------------
# drifts, bounds, standard deviations
# -----------------------------------------------------------------------------

# Drift

abstract AbstractDMDrift
getdt(drift::AbstractDMDrift) = drift.dt

immutable ConstDMDrift <: AbstractDMDrift
    dt::Float64
    mu::Float64

    function ConstDMDrift(mu::Real, dt::Real)
        dt > zero(dt) || error("dt needs to be positive")
        new(float(dt), float(mu))
    end
end
getmu(d::ConstDMDrift, n::Int) = d.mu
getm(d::ConstDMDrift, n::Int) = (n-1) * d.dt * d.mu
getmaxn(d::ConstDMDrift) = typemax(Int)

immutable DMDrift <: AbstractDMDrift
    dt::Float64
    mu::Vector{Float64}
    m::Vector{Float64}

    function DMDrift(mu::Vector{Float64}, dt::Real)
        dt > zero(dt) || error("dt needs to be positive")
        length(mu) > 0 || error("mu needs to be of non-zero length")
        new(float(dt), mu, [0.0, cumsum(mu[1:(end-1)]) * float(dt)])
    end
    function DMDrift(mu::Vector{Float64}, dt::Real, maxt::Real)
        dt > zero(dt) || error("dt needs to be positive")
        maxt >= dt || error("maxt needs to be at least as large as dt")
        n = length(0:dt:maxt)
        nmu = length(mu)
        nmu < n ? DMDrift([mu, fill(mu[end], n - nmu)], dt) : DMDrift(mu[1:n], dt)
    end
end
getmu(d::DMDrift, n::Int) = d.mu[n]
getm(d::DMDrift, n::Int) = d.m[n]
getmaxn(d::DMDrift) = length(d.mu)

# Bounds

abstract AbstractDMBounds
getdt(bounds::AbstractDMBounds) = bounds.dt

immutable ConstSymDMBounds <: AbstractDMBounds
    dt::Float64
    b::Float64

    function ConstSymDMBounds(b::Real, dt::Real)
        b > zero(b) || error("bound needs to be positive")
        dt > zero(dt) || error("dt needs to be positive")
        new(float(dt), float(b))
    end
end
getb1(b::ConstSymDMBounds, n::Int) = b.b
getb2(b::ConstSymDMBounds, n::Int) = -b.b
getb1g(b::ConstSymDMBounds, n::Int) = 0.0
getb2g(b::ConstSymDMBounds, n::Int) = 0.0
getmaxn(b::ConstSymDMBounds) = typemax(Int)


immutable ConstDMBounds <: AbstractDMBounds
    dt::Float64
    b1::Float64
    b2::Float64

    function ConstDMBounds(b1::Real, b2::Real, dt::Real)
        b1 > zero(b1) || error("b1 needs to be positive")
        b2 < zero(b2) || error("b2 needs to be negative")
        dt > zero(dt) || error("dt needs to be positive")
        new(float(dt), float(b1), float(b2))
    end
end
getb1(b::ConstDMBounds, n::Int) = b.b1
getb2(b::ConstDMBounds, n::Int) = b.b2
getb1g(b::ConstDMBounds, n::Int) = 0.0
getb2g(b::ConstDMBounds, n::Int) = 0.0
getmaxn(b::ConstDMBounds) = typemax(Int)


immutable DMBounds <: AbstractDMBounds
    dt::Float64
    b1::Vector{Float64}
    b2::Vector{Float64}
    b1g::Vector{Float64}
    b2g::Vector{Float64}

    function DMBounds(b1::Vector{Float64}, b2::Vector{Float64},
        b1g::Vector{Float64}, b2g::Vector{Float64}, dt::Real)
        dt > zero(dt) || error("dt needs to be positive")
        length(b1) > 0 || error("bounds need to be of non-zero length")
        length(b1) == length(b2) == length(b1g) == length(b2g) ||
            error("bounds need to be of same length")
        b1[1] > 0.0 || error("b1[1] needs to be positive")
        b2[1] < 0.0 || error("b2[1] needs to be negative")
        new(dt, b1, b2, b1g, b2g)
    end
    DMBounds(b1::Vector{Float64}, b2::Vector{Float64}, dt::Real) =
        DMBounds(b1, b2, fdgrad(b1, dt), fdgrad(b2, dt), dt)
end
getb1(b::DMBounds, n::Int) = b.b1[n]
getb2(b::DMBounds, n::Int) = b.b2[n]
getb1g(b::DMBounds, n::Int) = b.b1g[n]
getb2g(b::DMBounds, n::Int) = b.b2g[n]
getmaxn(b::DMBounds) = length(b.b1)


immutable SymDMBounds <: AbstractDMBounds
    dt::Float64
    b::Vector{Float64}
    bg::Vector{Float64}

    function SymDMBounds(b::Vector{Float64}, bg::Vector{Float64}, dt::Real)
        dt > zero(dt) || error("dt needs to be positive")
        length(b) > 0 || error("bound needs to be of non-zero length")
        length(b) == length(bg) || error("bounds need to be of same length")
        b[1] > 0.0 || error("b[1] needs to be positive")
        new(dt, b, bg)
    end
    SymDMBounds(b::Vector{Float64}, dt::Real) = SymDMBounds(b, fdgrad(b, dt), dt)
end
getb1(b::SymDMBounds, n::Int) = b.b[n]
getb2(b::SymDMBounds, n::Int) = -b.b[n]
getb1g(b::SymDMBounds, n::Int) = b.bg[n]
getb2g(b::SymDMBounds, n::Int) = -b.bg[n]
getmaxn(b::SymDMBounds) = length(b.b)


# -----------------------------------------------------------------------------
# first-passage-time distributions
# -----------------------------------------------------------------------------

# algorithm adapted from 
#    DJ Navarro & IG Fuss (2009). Fast and accurate calculations for
#    first-passage times in Wiener diffusion models. Journal of Mathematical
#    Psychology, 53(4), 222-230.
function fptdist(d::ConstDMDrift, b::ConstSymDMBounds, tmax::Real, tol::Real=1.e-29)
    tmax >= zero(tmax) || error("tmax needs to be non-negative")

    dt = getdt(d)
    @assert getdt(b) == dt
    mu, bound = d.mu, b.b
    maxn = length(0:dt:tmax)

    g1, g2 = Array(Float64, maxn), Array(Float64, maxn)
    g1[1], g2[1] = 0.0, 0.0
    const c1, c2 = 4bound * bound, abs2(mu)
    const c3, c4 = exp(mu * bound) / c1, exp(-2mu * bound)
    t = dt
    for n = 2:maxn
        g1[n] = c3 * exp(-c2 * t / 2) * ftpdist_fastseries(t / c1, 0.5, tol)
        g2[n] = c4 * g1[n]
        t += dt
    end
    g1, g2
end

# algorithm adapted from
#    DJ Navarro & IG Fuss (2009). Fast and accurate calculations for
#    first-passage times in Wiener diffusion models. Journal of Mathematical
#    Psychology, 53(4), 222-230.
function fptdist(d::ConstDMDrift, b::ConstDMBounds, tmax::Real, tol::Real=1.e-29)
    tmax >= zero(tmax) || error("tmax needs to be non-negative")

    dt = getdt(d)
    @assert getdt(b) == dt
    mu, b1, b2 = d.mu, b.b1, b.b2
    maxn = length(0:dt:tmax)

    g1, g2 = Array(Float64, maxn), Array(Float64, maxn)
    g1[1], g2[1] = 0.0, 0.0
    const c1, c2, w = abs2(b1 - b2), abs2(mu), -b2 / (b1 - b2)
    const c3, c4 = exp(mu * b1) / c1, exp(mu * b2) / c1
    t = dt
    for n = 2:maxn
        const mu2t, t_scaled = exp(-c2 * t / 2), t / c1
        g1[n] = c3 * mu2t * ftpdist_fastseries(t_scaled, 1.0 - w, tol)
        g2[n] = c4 * mu2t * ftpdist_fastseries(t_scaled, w, tol)
        t += dt
    end
    g1, g2
end

# fpt density for mu=0, constant bounds at -0.5 and 0.5, and starting point 
# at w, using series expansion appropriate for given t.
# Impements Navarro & Fuss (2009), Eq. (13)
function ftpdist_fastseries(t::Float64, w::Float64, tol::Float64)
    const Ksin = sqrt(-2log(π * t * tol) / (π * π * t))
    const Kexp = 2 + sqrt(-2t * log(2tol * sqrt(twoπ * t)))
    Kexp < Ksin ? ftpdist_expseries(t, w, tol) : ftpdist_sinseries(t, w, tol)
end

# fpt density for mu=0, constant bounds at -0.5 and 0.5, and starting point 
# at w, using series expansion that is accurate/fast for small t.
# Implements Navarro & Fuss (2009), Eq. (6)
function ftpdist_expseries(t::Float64, w::Float64, tol::Float64)
    f = w * exp(-w * w / 2t)
    k = 1
    while true
        c = w + 2k
        incr = c * exp(-c * c / 2t)
        f += incr
        tol < abs(incr) || break
        c = w - 2k
        incr = c * exp(-c * c / 2t)
        f += incr
        tol < abs(incr) || break
        k += 1
    end
    f * t^-1.5 / sqrt2π
end

# fpt density for mu=0, constant bounds at -0.5 and 0.5, and starting point 
# at w, using series expansion that is accurate/fast for large t
# Implements Navarro & Fuss (2009), Eq. (5)
function ftpdist_sinseries(t::Float64, w::Float64, tol::Float64)
    f, k = 0.0, 1
    while true
        incr = k * exp(-abs2(k * π) * t / 2) * sinpi(k * w)
        f += incr
        tol < abs(incr) || break
        k += 1
    end
    f * π
end

function fptdist(d::DMDrift, b::AbstractDMBounds, tmax::Real)
    tmax >= zero(tmax) || error("tmax needs to be non-negative")

    dt = getdt(d)
    @assert getdt(b) == dt
    maxn = length(0:dt:tmax)
    @assert getmaxn(d) >= maxn && getmaxn(b) >= maxn

    # Volterra series expansion from
    #     PL Smith (2000). Stochastic dynamic models of response time and
    #     accuracy: a foundational primer. Journal of Mathematical
    #     psychology 44, 408-463.
    g1, g2 = Array(Float64, maxn), Array(Float64, maxn)
    g1[1], g2[1] = 0.0, 0.0
    if maxn == 1
        return g1, g2
    end
    const c1 = 1.0 / sqrt(twoπ * dt)
    b1d, b2d = getb1(b, 2) - getm(d, 2), getb2(b, 2) - getm(d, 2)
    g1[2] = - c1 * exp(- b1d * b1d / 2dt) * 
        (getb1g(b, 2) - getmu(d, 2) - b1d / dt)
    g2[2] = c1 * exp(- b2d * b2d / 2dt) *
        (getb2g(b, 2) - getmu(d, 2) - b2d / dt)
    for n = 3:maxn
        g1n, g2n = 0.0, 0.0
        const mun, mn = getmu(d, n), getm(d, n)
        const bupn, blon = getb1(b, n), getb2(b, n)
        const bupgradn, blogradn = getb1g(b, n), getb2g(b, n)
        for j = 2:(n-1)
            b1d = bupn - getb1(b, j) + getm(d, j) - mn
            b2d = bupn - getb2(b, j) + getm(d, j) - mn
            g1n += c1 / sqrt(n-j) * (
                g1[j] * exp(- b1d * b1d / (2dt * (n-j))) * 
                (bupgradn - mun - b1d / (dt * (n-j))) +
                g2[j] * exp(- b2d * b2d / (2dt * (n-j))) *
                (bupgradn - mun - b2d / (dt * (n-j))))
            b1d = blon - getb1(b, j) + getm(d, j) - mn
            b2d = blon - getb2(b, j) + getm(d, j) - mn
            g2n += c1 / sqrt(n-j) * (
                g1[j] * exp(- b1d * b1d / (2dt * (n-j))) *
                (blogradn - mun - b1d / (dt * (n-j))) +
                g2[j] * exp(- b2d * b2d / (2dt * (n-j))) *
                (blogradn - mun - b2d / (dt * (n-j))))
        end
        b1d, b2d = bupn - mn, blon - mn
        g1[n] = - c1 / sqrt(n-1) * exp(- b1d * b1d / (2dt * (n-1))) *
            (bupgradn - mun - b1d / (dt * (n-1))) + dt * g1n
        g2[n] = c1 / sqrt(n-1) * exp(- b2d * b2d / (2dt * (n-1))) *
            (blogradn - mun - b2d / (dt * (n-1))) - dt * g2n
    end
    g1, g2
end