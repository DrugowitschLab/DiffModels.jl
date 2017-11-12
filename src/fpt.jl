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

abstract type PDFConstCacheBase end

struct AsymPDFConstCache <: PDFConstCacheBase
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

pdful(c::PDFConstCacheBase, t::Real, tol::Real=1.e-29) =
    pdfu(c, t, tol), pdfl(c, t, tol)
pdfu(c::AsymPDFConstCache, t::Real, tol::Real=1.e-29) =
    exp(c.c3 - c.c2 * t) / c.c1 * pdf_asymfastseries(t / c.c1, 1 - c.w, tol)
pdfl(c::AsymPDFConstCache, t::Real, tol::Real=1.e-29) =
    exp(c.c4 - c.c2 * t) / c.c1 * pdf_asymfastseries(t / c.c1, c.w, tol)

# fpt density for mu=0, constant bounds at 0 and 1, and starting point at w
pdf_asymfastseries(t::Real, w::Real, tol::Real) = 
    t == 0.0 ? 0.0 :
        useshorttseries(t, tol) ? pdf_asymshortt(t, w, tol) : 
                                  pdf_asymlongt(t, w, tol)

# Navarro & Fuss (2009), Eq. (6)
function pdf_asymshortt(t::Real, w::Real, tol::Real)
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
function pdf_asymlongt(t::Real, w::Real, tol::Real)
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

struct SymPDFConstCache <: PDFConstCacheBase
    c1::Float64
    c2::Float64
    c3::Float64

    function SymPDFConstCache(d::ConstDrift, b::ConstSymBounds)
        const mu, bound = getmu(d), getbound(b)
        new(4 * abs2(bound), abs2(mu) / 2, mu * bound)
    end
end

function pdful(c::SymPDFConstCache, t::Real, tol::Real=1.e-29)
    const g = pdf_symfastseries(t / c.c1, tol) / c.c1
    exp(c.c3 - c.c2 * t) * g, exp(- c.c3 - c.c2 * t) * g
end
pdfu(c::SymPDFConstCache, t::Real, tol::Real=1.e-29) =
    exp(c.c3 - c.c2 * t) / c.c1 * pdf_symfastseries(t / c.c1, tol)
pdfl(c::SymPDFConstCache, t::Real, tol::Real=1.e-29) =
    exp(- c.c3 - c.c2 * t) / c.c1 * pdf_symfastseries(t / c.c1, tol)

# fpt density for mu=0, constant bounds at 0 and 1, and starting point at 0.5,
pdf_symfastseries(t::Real, tol::Real) =
    t == 0.0 ? 0.0 :
        useshorttseries(t, tol) ?
            pdf_symseries(t, 1 / 8t, 1 / sqrt(8 * π * t^3), tol) :
            pdf_symseries(t, t * abs2(π) / 2, float(π), tol)

function pdf_symseries(t::Real, a::Real, b::Real, tol::Real)
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

pdfu(d::ConstDrift, b::ConstBounds, t::Real, tol::Real=1.e-29) = pdfu(PDFConstCache(d, b), t, tol)
pdfl(d::ConstDrift, b::ConstBounds, t::Real, tol::Real=1.e-29) = pdfl(PDFConstCache(d, b), t, tol)
pdful(d::ConstDrift, b::ConstBounds, t::Real, tol::Real=1.e-29) = pdful(PDFConstCache(d, b), t, tol)

function pdf(d::ConstDrift, b::ConstBounds, tmax::Real, tol::Real=1.e-29)
    tmax >= zero(tmax) || error("tmax needs to be non-negative")
    const c = PDFConstCache(d, b)
    const dt = getdt(d)
    @assert getdt(b) == dt
    const maxn = length(0:dt:tmax)

    gu, gl, t = Array{Float64}(maxn), Array{Float64}(maxn), 0.0
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

    g1, g2 = Array{Float64}(maxn), Array{Float64}(maxn)
    g1[1], g2[1] = 0.0, 0.0
    if maxn == 1
        return g1, g2
    end

    const cΨ = 1.0 / √(twoπ)
    # computes diffusion kernel 2Ψ(b(t), t | y, tau) for parameters
    # mut = drift(t), bgradt = d b(t) / dt, dbm = b(t) - y - m(t) + m(tau),
    # dv = v(t) - v(tau), sqrtdv = √(dv)
    _Ψ(mut, bgradt, dbm, dv, sqrtdv) =
        cΨ * exp(-dbm * dbm / (2dv)) / sqrtdv * (bgradt - mut - dbm / dv)

    g1[2] = - _Ψ(getmu(d, 2), getuboundgrad(b, 2), getubound(b, 2) - getm(d, 2), dt, √(dt))
    g2[2] = _Ψ(getmu(d, 2), getlboundgrad(b, 2), getlbound(b, 2) - getm(d, 2), dt, √(dt))

    for n = 3:maxn
        g1n, g2n = 0.0, 0.0
        const mun, mn = getmu(d, n), getm(d, n)
        const bupn, blon = getubound(b, n), getlbound(b, n)
        const bupgradn, blogradn = getuboundgrad(b, n), getlboundgrad(b, n)
        for j = 2:(n-1)
            const dv = dt * (n-j)
            const sqrtdv = √(dv)
            const dbmu = getm(d, j) - mn - getubound(b, j)
            const dbml = getm(d, j) - mn - getlbound(b, j)
            g1n += g1[j] * _Ψ(mun, bupgradn, bupn + dbmu, dv, sqrtdv) +
                g2[j] * _Ψ(mun, bupgradn, bupn + dbml, dv, sqrtdv)
            g2n += g1[j] * _Ψ(mun, blogradn, blon + dbmu, dv, sqrtdv) +
                g2[j] * _Ψ(mun, blogradn, blon + dbml, dv, sqrtdv)
        end
        g1[n] = - _Ψ(mun, bupgradn, bupn - mn, dt * (n-1), √(dt * (n-1))) + dt * g1n
        g2[n] = _Ψ(mun, blogradn, blon - mn, dt * (n-1), √(dt * (n-1))) - dt * g2n
    end
    g1, g2
end
