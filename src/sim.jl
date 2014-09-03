# diffusion model samples

type DMBoundOutcome  <: ValueSupport end
type DMBoundsOutcome <: ValueSupport end

typealias DMBoundSampleable  Sampleable{Univariate, DMBoundOutcome}
typealias DMBoundsSampleable Sampleable{Multivariate, DMBoundsOutcome}

Base.eltype(::Type{DMBoundOutcome}) = Float64
Base.eltype(::Type{DMBoundsOutcome}) = Float64, Bool

Base.length(::DMBoundSampleable) = 1
Base.length(::DMBoundsSampleable) = 2

sampler(d::AbstractDrift, b::AbstractBounds) = DMBoundsSampler(d, b)

immutable DMBoundsSampler
    d::AbstractDrift
    b::AbstractBounds
    sqrtdt::Float64

    function DMBoundsSampler(d::AbstractDrift, b::AbstractBounds)
        const dt = getdt(d)
        @assert dt == getdt(b)
        new(d, b, sqrt(dt))
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
