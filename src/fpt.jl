# First-passage time distributions for diffusion models

# TODO:
# - add version that includes leak
# - add version that includes time-varying variance

# constant bound, constant drift
#
# algorithms adapted from 
#    DJ Navarro & IG Fuss (2009). Fast and accurate calculations for
#    first-passage times in Wiener diffusion models. Journal of Mathematical
#    Psychology, 53(4), 222-230.

# Navarro & Fuss (2009), Eq. (13)
useshorttseries(t::Real, tol::Real) =
  2 + sqrt(-2t * log(2tol * sqrt(twoπ * t))) <
  sqrt(-2log(π * t * tol) / (t * abs2(π)))

abstract PDFConstCacheBase

immutable AsymPDFConstCache <: PDFConstCacheBase
    c1::Float64
    c2::Float64
    c3::Float64
    c4::Float64
    w::Float64

    function AsymPDFConstCache(d::ConstDrift, b::ConstAsymBounds)
        const mu, bu, bl = getmu(d), getubound(b), getlbound(b)
        new(abs2(bu - bl), abs2(mu) / 2, mu * bu, mu * bl, -bl / (bu - bl))
    end
end

pdful(c::PDFConstCacheBase, t::Float64, tol::Real=1.e-29) =
    pdfu(c, t, tol), pdfl(c, t, tol)
pdfu(c::AsymPDFConstCache, t::Float64, tol::Real=1.e-29) =
    exp(c.c3 - c.c2 * t) / c.c1 * pdf_asymfastseries(t / c.c1, 1 - c.w, tol)
pdfl(c::AsymPDFConstCache, t::Float64, tol::Real=1.e-29) =
    exp(c.c4 - c.c2 * t) / c.c1 * pdf_asymfastseries(t / c.c1, c.w, tol)

# fpt density for mu=0, constant bounds at 0 and 1, and starting point at w
pdf_asymfastseries(t::Float64, w::Float64, tol::Float64) = 
    t == 0.0 ? 0.0 :
        useshorttseries(t, tol) ? pdf_asymshortt(t, w, tol) : 
                                  pdf_asymlongt(t, w, tol)

# Navarro & Fuss (2009), Eq. (6)
function pdf_asymshortt(t::Float64, w::Float64, tol::Float64)
    const b = t^-1.5 / sqrt2π
    tol *= b
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
    f * b
end

# Navarro & Fuss (2009), Eq. (5)
function pdf_asymlongt(t::Float64, w::Float64, tol::Float64)
    tol *= π
    f, k = 0.0, 1
    while true
        incr = k * exp(-abs2(k * π) * t / 2) * sinpi(k * w)
        f += incr
        tol < abs(incr) || break
        k += 1
    end
    f * π
end

immutable SymPDFConstCache <: PDFConstCacheBase
    c1::Float64
    c2::Float64
    c3::Float64

    function SymPDFConstCache(d::ConstDrift, b::ConstSymBounds)
        const mu, bound = getmu(d), getbound(b)
        new(4 * abs2(bound), abs2(mu) / 2, mu * bound)
    end
end

function pdful(c::SymPDFConstCache, t::Float64, tol::Real=1.e-29)
    const g = pdf_symfastseries(t / c.c1, tol) / c.c1
    exp(c.c3 - c.c2 * t) * g, exp(- c.c3 - c.c2 * t) * g
end
pdfu(c::SymPDFConstCache, t::Float64, tol::Real=1.e-29) =
    exp(c.c3 - c.c2 * t) / c.c1 * pdf_symfastseries(t / c.c1, tol)
pdfl(c::SymPDFConstCache, t::Float64, tol::Real=1.e-29) =
    exp(- c.c3 - c.c2 * t) / c.c1 * pdf_symfastseries(t / c.c1, tol)

# fpt density for mu=0, constant bounds at 0 and 1, and starting point at 0.5,
pdf_symfastseries(t::Float64, tol::Float64) =
    t == 0.0 ? 0.0 :
        useshorttseries(t, tol) ?
            pdf_symseries(t, 1 / 8t, 1 / sqrt(8 * π * t^3), tol) :
            pdf_symseries(t, t * abs2(π) / 2, float(π), tol)

function pdf_symseries(t::Float64, a::Float64, b::Float64, tol::Float64)
    tol *= b
    f, twok = exp(-a), 3
    while true
        incr = twok * exp(- abs2(twok) * a)
        f -= incr
        tol < incr || break
        twok += 2
        incr = twok * exp(- abs2(twok) * a)
        f += incr
        tol < incr || break
        twok += 2
    end
    b * f
end

PDFConstCache(d::ConstDrift, b::ConstAsymBounds) = AsymPDFConstCache(d, b)
PDFConstCache(d::ConstDrift, b::ConstSymBounds) = SymPDFConstCache(d, b)

pdfu(d::ConstDrift, b::ConstBounds, t::Float64, tol::Real=1.e-29) = pdfu(PDFConstCache(d, b), t, tol)
pdfl(d::ConstDrift, b::ConstBounds, t::Float64, tol::Real=1.e-29) = pdfl(PDFConstCache(d, b), t, tol)
pdful(d::ConstDrift, b::ConstBounds, t::Float64, tol::Real=1.e-29) = pdful(PDFConstCache(d, b), t, tol)

function pdf(d::ConstDrift, b::ConstBounds, tmax::Real, tol::Real=1.e-29)
    tmax >= zero(tmax) || error("tmax needs to be non-negative")
    const c = PDFConstCache(d, b)
    const dt = getdt(d)
    @assert getdt(b) == dt
    const maxn = length(0:dt:tmax)

    gu, gl, t = Array(Float64, maxn), Array(Float64, maxn), 0.0
    for n = 1:maxn
        gu[n], gl[n] = pdful(c, t, tol)
        t += dt
    end
    gu, gl
end


# time-varying bound, time-varying drift
#
# using Volterra series expansion from
#     PL Smith (2000). Stochastic dynamic models of response time and
#     accuracy: a foundational primer. Journal of Mathematical
#     psychology 44, 408-463.

function pdf(d::AbstractDrift, b::AbstractBounds, tmax::Real)
    tmax >= zero(tmax) || error("tmax needs to be non-negative")

    dt = getdt(d)
    @assert getdt(b) == dt
    maxn = length(0:dt:tmax)
    @assert getmaxn(d) >= maxn && getmaxn(b) >= maxn

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
