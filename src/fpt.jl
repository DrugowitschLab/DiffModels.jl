# First-passage time distributions for diffusion models

# TODO:
# - add version that includes leak
# - add version that includes time-varying variance

# algorithm adapted from 
#    DJ Navarro & IG Fuss (2009). Fast and accurate calculations for
#    first-passage times in Wiener diffusion models. Journal of Mathematical
#    Psychology, 53(4), 222-230.
function pdf(d::ConstDrift, b::ConstSymBounds, tmax::Real, tol::Real=1.e-29)
    tmax >= zero(tmax) || error("tmax needs to be non-negative")

    const dt = getdt(d)
    @assert getdt(b) == dt
    const mu, bound = getmu(d), getbound(b)
    maxn = length(0:dt:tmax)

    g1, g2 = Array(Float64, maxn), Array(Float64, maxn)
    g1[1], g2[1] = 0.0, 0.0
    const c1, c2 = 4bound * bound, abs2(mu)
    const c3, c4 = exp(mu * bound) / c1, exp(-2mu * bound)
    t = dt
    for n = 2:maxn
        g1[n] = c3 * exp(-c2 * t / 2) * pdf_fastseries(t / c1, 0.5, tol)
        g2[n] = c4 * g1[n]
        t += dt
    end
    g1, g2
end


# algorithm adapted from
#    DJ Navarro & IG Fuss (2009). Fast and accurate calculations for
#    first-passage times in Wiener diffusion models. Journal of Mathematical
#    Psychology, 53(4), 222-230.
function pdf(d::ConstDrift, b::ConstAsymBounds, tmax::Real, tol::Real=1.e-29)
    tmax >= zero(tmax) || error("tmax needs to be non-negative")

    const dt = getdt(d)
    @assert getdt(b) == dt
    const mu, bu, bl = getmu(d), getubound(b), getlbound(b)
    const maxn = length(0:dt:tmax)

    g1, g2 = Array(Float64, maxn), Array(Float64, maxn)
    g1[1], g2[1] = 0.0, 0.0
    const c1, c2, w = abs2(bu - bl), abs2(mu), -bl / (bu - bl)
    const c3, c4 = exp(mu * bu) / c1, exp(mu * bl) / c1
    t = dt
    for n = 2:maxn
        const mu2t, t_scaled = exp(-c2 * t / 2), t / c1
        g1[n] = c3 * mu2t * pdf_fastseries(t_scaled, 1.0 - w, tol)
        g2[n] = c4 * mu2t * pdf_fastseries(t_scaled, w, tol)
        t += dt
    end
    g1, g2
end


# fpt density for mu=0, constant bounds at 0 and 1, and starting point at w,
# using series expansion appropriate for given t.
# Impements Navarro & Fuss (2009), Eq. (13)
function pdf_fastseries(t::Float64, w::Float64, tol::Float64)
    const Ksin = sqrt(-2log(π * t * tol) / (π * π * t))
    const Kexp = 2 + sqrt(-2t * log(2tol * sqrt(twoπ * t)))
    Kexp < Ksin ? pdf_expseries(t, w, tol) : pdf_sinseries(t, w, tol)
end


# fpt density for mu=0, constant bounds at 0 and 1, and starting point at w,
# using series expansion that is accurate/fast for small t.
# Implements Navarro & Fuss (2009), Eq. (6)
function pdf_expseries(t::Float64, w::Float64, tol::Float64)
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


# fpt density for mu=0, constant bounds at 0 and 1, and starting point at w,
# using series expansion that is accurate/fast for large t
# Implements Navarro & Fuss (2009), Eq. (5)
function pdf_sinseries(t::Float64, w::Float64, tol::Float64)
    f, k = 0.0, 1
    while true
        incr = k * exp(-abs2(k * π) * t / 2) * sinpi(k * w)
        f += incr
        tol < abs(incr) || break
        k += 1
    end
    f * π
end


function pdf(d::AbstractDrift, b::AbstractBounds, tmax::Real)
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
    b1d, b2d = getubound(b, 2) - getm(d, 2), getlbound(b, 2) - getm(d, 2)
    g1[2] = - c1 * exp(- b1d * b1d / 2dt) * 
        (getuboundgrad(b, 2) - getmu(d, 2) - b1d / dt)
    g2[2] = c1 * exp(- b2d * b2d / 2dt) *
        (getlboundgrad(b, 2) - getmu(d, 2) - b2d / dt)
    for n = 3:maxn
        g1n, g2n = 0.0, 0.0
        const mun, mn = getmu(d, n), getm(d, n)
        const bupn, blon = getubound(b, n), getlbound(b, n)
        const bupgradn, blogradn = getuboundgrad(b, n), getlboundgrad(b, n)
        for j = 2:(n-1)
            b1d = bupn - getubound(b, j) + getm(d, j) - mn
            b2d = bupn - getlbound(b, j) + getm(d, j) - mn
            g1n += c1 / sqrt(n-j) * (
                g1[j] * exp(- b1d * b1d / (2dt * (n-j))) * 
                (bupgradn - mun - b1d / (dt * (n-j))) +
                g2[j] * exp(- b2d * b2d / (2dt * (n-j))) *
                (bupgradn - mun - b2d / (dt * (n-j))))
            b1d = blon - getubound(b, j) + getm(d, j) - mn
            b2d = blon - getlbound(b, j) + getm(d, j) - mn
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
