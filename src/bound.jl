# constant and time-varying bounds and bound pairs

# TODO:
# - add bounds constructors with maxt, to extend/shorten given bounds

# single bounds

abstract type AbstractBound end
getdt(b::AbstractBound) = b.dt

struct ConstBound <: AbstractBound
    b::Float64
    dt::Float64

    function ConstBound(b::Real, dt::Real)
        b > zero(b) || error("bound needs to be positive")
        dt > zero(dt) || error("dt needs to be positive")
        return new(float(b), float(dt))
    end
end
getbound(b::ConstBound, n::Int) = b.b
getbound(b::ConstBound) = b.b
getboundgrad(b::ConstBound, n::Int) = 0.0
getmaxn(b::ConstBound) = typemax(Int)

struct VarBound <: AbstractBound
    b::Vector{Float64}
    bg::Vector{Float64}
    dt::Float64

    function VarBound(b::AbstractVector{T1}, bg::AbstractVector{T2},
                      dt::Real) where {T1 <: Real, T2 <: Real}
        dt > zero(dt) || error("dt needs to be positive")
        length(b) > 0 || error("bounds need to be of non-zero length")
        length(b) == length(bg) || error("b and bg need to be of same length")
        b[1] > 0.0 || error("b[1] needs to be positive")
        return new(b, bg, float(dt))
    end
    VarBound(b::AbstractVector{T}, dt::Real) where T <: Real = VarBound(b, fdgrad(b, dt), dt)
end
getbound(b::VarBound, n::Int) = b.b[n]
getboundgrad(b::VarBound, n::Int) = b.bg[n]
getmaxn(b::VarBound) = length(b.b)

struct LinearBound <: AbstractBound
    b0::Float64
    bslope::Float64
    dtbslope::Float64
    dt::Float64
    maxn::Int

    function LinearBound(b0::Real, bslope::Real, dt::Real, maxt::Real=Inf)
        dt > zero(dt) || error("dt needs to be positive")
        b0 > zero(dt) || error("b0 needs to be positive")
        maxt > zero(maxt) || error("maxt needs to be positive")
        maxn = maxt / dt
        # subtracting dt * bslope ensures that getbound(b, 1) = b0
        return new(b0 - dt * bslope, bslope, dt * bslope, dt,
            isfinite(maxn) ? ceil(Int, maxn) : typemax(Int))
    end
end
getbound(b::LinearBound, n::Int) = b.b0 + n * b.dtbslope
getboundgrad(b::LinearBound, n::Int) = b.bslope
getmaxn(b::LinearBound) = b.maxn


# bound pairs

abstract type AbstractBounds end

struct SymBounds{T <: AbstractBound} <: AbstractBounds
    b::T

    SymBounds{T}(b::T) where T = new(b)
end
getdt(b::SymBounds) = getdt(b.b)
getubound(b::SymBounds, n::Int) = getbound(b.b, n)
getlbound(b::SymBounds, n::Int) = -getbound(b.b, n)
getuboundgrad(b::SymBounds, n::Int) = getboundgrad(b.b, n)
getlboundgrad(b::SymBounds, n::Int) = -getboundgrad(b.b, n)
getmaxn(b::SymBounds) = getmaxn(b.b)

const VarSymBounds = SymBounds{VarBound}

const ConstSymBounds = SymBounds{ConstBound}
ConstSymBounds(b::Real, dt::Real) = ConstSymBounds(ConstBound(b, dt))
getbound(b::ConstSymBounds) = b.b.b

const LinearSymBounds = SymBounds{LinearBound}
LinearSymBounds(b0::Real, bslope::Real, dt::Real) = LinearSymBounds(LinearBound(b0, bslope, dt))

struct AsymBounds{T1 <: AbstractBound, T2 <: AbstractBound} <: AbstractBounds
    upper::T1
    lower::T2

    function AsymBounds{T1,T2}(upper::T1, lower::T2) where {T1,T2}
        getdt(upper) == getdt(lower) || error("Bounds need to have equal dt")
        return new(upper, lower)
    end
end
getdt(b::AsymBounds) = getdt(b.upper)
getubound(b::AsymBounds, n::Int) = getbound(b.upper, n)
getlbound(b::AsymBounds, n::Int) = -getbound(b.lower, n)
getuboundgrad(b::AsymBounds, n::Int) = getboundgrad(b.upper, n)
getlboundgrad(b::AsymBounds, n::Int) = -getboundgrad(b.lower, n)
getmaxn(b::AsymBounds) = min(getmaxn(b.upper), getmaxn(b.lower))

const VarAsymBounds = AsymBounds{VarBound, VarBound}

const ConstAsymBounds = AsymBounds{ConstBound, ConstBound}
ConstAsymBounds(upper::Real, lower::Real, dt::Real) =
    ConstAsymBounds(ConstBound(upper, dt), ConstBound(-lower, dt))
getubound(b::ConstAsymBounds) = getbound(b.upper)
getlbound(b::ConstAsymBounds) = -getbound(b.lower)

const ConstBounds = Union{ConstSymBounds, ConstAsymBounds}
