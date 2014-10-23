# diffusion model samples

type DMBoundOutcome  <: ValueSupport end
type DMBoundsOutcome <: ValueSupport end

typealias DMBoundSampleable  Sampleable{Univariate, DMBoundOutcome}
typealias DMBoundsSampleable Sampleable{Multivariate, DMBoundsOutcome}

Base.eltype(::Type{DMBoundOutcome}) = Float64
Base.eltype(::Type{DMBoundsOutcome}) = Float64, Bool

Base.length(::DMBoundSampleable) = 1
Base.length(::DMBoundsSampleable) = 2

# general Euler-Maruyama sampler

sampler(d::AbstractDrift, b::AbstractBounds) = DMBoundsSampler(d, b)

immutable DMBoundsSampler
    d::AbstractDrift
    b::AbstractBounds
    sqrtdt::Float64

    function DMBoundsSampler(d::AbstractDrift, b::AbstractBounds)
        const dt = getdt(d)
        @assert dt == getdt(b)
        new(d, b, √(dt))
    end
end

function rand(s::DMBoundsSampler)
    x, n, dt, maxn = 0.0, 1, getdt(s.d), min(getmaxn(s.d), getmaxn(s.b))
    while n < maxn
        x += getmu(s.d, n) * dt + s.sqrtdt * randn()
        n += 1
        if x >= getubound(s.b, n)
            return (n - 1) * dt, true
        elseif x <= getlbound(s.b, n)
            return (n - 1) * dt, false
        end
    end
    # no bound crossing until n - return random sample and Inf
    Inf, rand() < 0.5 
end

# faster sampler for constant bounds

# samples only fpt, assumes bounds at -1, 1, unit variance
abstract DMConstSymBoundsFPTSampler

immutable DMConstSymBoundsNormExpSampler <: DMConstSymBoundsFPTSampler
    mu2::Float64
    a::Float64
    sqrtamu::Float64
    fourmu2π::Float64
    Cf1s::Float64
    CF1st::Float64
    Cf1l::Float64
    F1lt::Float64
    F1inf::Float64

    function DMConstSymBoundsNormExpSampler(mu::Float64)
        const mu2 = abs2(mu)
        const tthresh = 0.12+0.5exp(-abs(mu)/3)
        const a = (3 + √(9 + 4mu2)) / 6
        const sqrtamu = √((a-1) * mu2 / a)
        const fourmu2π = (4mu2 + abs2(π)) / 8
        const Cf1s = √(a) * exp(-sqrtamu)
        const Cf1l = π / 4fourmu2π
        const CF1st = Cf1s * erfc(1 / √(2a * tthresh))
        const F1lt = - expm1(-tthresh * fourmu2π)
        const F1inf = CF1st + Cf1l * (1 - F1lt)
        new(mu2, a, sqrtamu, fourmu2π, Cf1s, CF1st, Cf1l, F1lt, F1inf)
    end
end

function rand(s::DMConstSymBoundsNormExpSampler)
    while true
        P = s.F1inf * rand()
        if P <= s.CF1st
            # short-time series
            t = 1 / (2s.a * abs2(erfcinv(P / s.Cf1s)))
            !_acceptt(t, exp(- 1/(2s.a*t) - s.sqrtamu + s.mu2*t/2), 1 / 2t) || return t
        else
            # long-time series
            t = -log1p(- (P - s.CF1st)/s.Cf1l - s.F1lt) / s.fourmu2π
            const π2t8 = abs2(π) * t / 8
            !_acceptt(t, exp(-π2t8), π2t8) || return t
        end
    end
end

immutable DMConstSymBoundsInvNormSampler <: DMConstSymBoundsFPTSampler
    invabsmu::Float64
    invmu2::Float64

    function DMConstSymBoundsInvNormSampler(mu::Float64)
        @assert mu > 0.0
        new(1 / abs(mu), 1 / abs2(mu))
    end
end

function rand(s::DMConstSymBoundsInvNormSampler)
    while true
        t = _invgaussrand(s.invabsmu, s.invmu2)
        const one2t = 1 / 2t
        if t < 2.5
            # short-time series
            !_acceptt(t, exp(-one2t), one2t) || return t
        else
            # long-time series
            const Cl = -log(π/4) - 0.5log(twoπ)
            !_acceptt(t, exp(Cl - one2t - 3log(t)/2), abs2(π) * t / 8) || return t
        end
    end
end

# sampler for inverse-Gamma distribution with lambda = 1, mean = mu, mu > 0
function _invgaussrand(mu, mu2)
    const y = abs2(randn())
    const x = mu + 0.5mu2 * y - 0.5mu * sqrt(4mu * y + mu2 * abs2(y))
    rand() <= 1 / (1 + x / mu) ? x : mu2 / x
end

function _acceptt(t, f, c2)
    @assert c2 > 0.06385320297074884 # log(5/3) / 16, req. for convergence
    const z = f * rand()
    b, twok = exp(-c2), 3
    while true
        z < b || return false
        b -= twok * exp(-c2 * abs2(twok))
        z > b || return true
        twok += 2
        b += twok * exp(-c2 * abs2(twok))
        twok += 2
    end
end

sampler(d::ConstDrift, b::ConstSymBounds) = DMConstSymBoundsSampler(d, b)

immutable DMConstSymBoundsSampler
    b2::Float64
    pu::Float64
    fpts::DMConstSymBoundsFPTSampler

    function DMConstSymBoundsSampler(d::ConstDrift, b::ConstSymBounds)
        const mu = getmu(d)
        const theta = getbound(b)
        const mutheta = theta * mu
        fpts = mutheta < 1.0 ?
               DMConstSymBoundsNormExpSampler(mutheta) :
               DMConstSymBoundsInvNormSampler(mutheta)
        new(abs2(theta), 1 / (1 + exp(-2mutheta)), fpts)
    end
end
rand(s::DMConstSymBoundsSampler) = rand(s.fpts) * s.b2, rand() < s.pu
