# constant and time-varying bounds and bound pairs

# TODO:
# - add bounds constructors with maxt, to extend/shorten given bounds

# single bounds

abstract AbstractBound
getdt(b::AbstractBound) = b.dt

immutable ConstBound <: AbstractBound
    b::Float64
    dt::Float64

    function ConstBound(b::Real, dt::Real)
        b > zero(b) || error("bound needs to be positive")
        dt > zero(dt) || error("dt needs to be positive")
        new(float(b), float(dt))
    end
end
getbound(b::ConstBound, n::Int) = b.b
getbound(b::ConstBound) = b.b
getboundgrad(b::ConstBound, n::Int) = 0.0
getmaxn(b::ConstBound) = typemax(Int)

immutable VarBound <: AbstractBound
    b::Vector{Float64}
    bg::Vector{Float64}
    dt::Float64

    function VarBound(b::Vector{Float64}, bg::Vector{Float64}, dt::Real)
        dt > zero(dt) || error("dt needs to be positive")
        length(b) > 0 || error("bounds need to be of non-zero length")
        length(b) == length(bg) || error("b and bg need to be of same length")
        b[1] > 0.0 || error("b[1] needs to be positive")
        new(b, bg, float(dt))
    end
    VarBound(b::Vector{Float64}, dt::Real) = VarBound(b, fdgrad(b, dt), dt)
end
getbound(b::VarBound, n::Int) = b.b[n]
getboundgrad(b::VarBound, n::Int) = b.bg[n]
getmaxn(b::VarBound) = length(b.b)

immutable LinearBound <: AbstractBound
    b0::Float64
    bslope::Float64
    dtbslope::Float64
    dt::Float64

    function LinearBound(b0::Real, bslope::Real, dt::Real)
        dt > zero(dt) || error("dt needs to be positive")
        b0 > zero(dt) || error("b0 needs to be positive")
        # subtracting dt * bslope ensures that getbound(b, 1) = b0
        new(b0 - dt * bslope, bslope, dt * bslope, dt)
    end
end
getbound(b::LinearBound, n::Int) = b.b0 + n * b.dtbslope
getboundgrad(b::LinearBound, n::Int) = b.bslope
getmaxn(b::LinearBound) = typemax(Int)


# bound pairs

abstract AbstractBounds

immutable SymBounds{T <: AbstractBound} <: AbstractBounds
    b::T

    SymBounds(b::T) = new(b)
end
getdt(b::SymBounds) = getdt(b.b)
getubound(b::SymBounds, n::Int) = getbound(b.b, n)
getlbound(b::SymBounds, n::Int) = -getbound(b.b, n)
getuboundgrad(b::SymBounds, n::Int) = getboundgrad(b.b, n)
getlboundgrad(b::SymBounds, n::Int) = -getboundgrad(b.b, n)
getmaxn(b::SymBounds) = getmaxn(b.b)

typealias VarSymBounds SymBounds{VarBound}

typealias ConstSymBounds SymBounds{ConstBound}
ConstSymBounds(b::Real, dt::Real) = ConstSymBounds(ConstBound(b, dt))
getbound(b::ConstSymBounds) = b.b.b

typealias LinearSymBounds SymBounds{LinearBound}
LinearSymBounds(b0::Real, bslope::Real, dt::Real) = LinearSymBounds(LinearBound(b0, bslope, dt))

immutable AsymBounds{T1 <: AbstractBound, T2 <: AbstractBound} <: AbstractBounds
    upper::T1
    lower::T2

    function AsymBounds(upper::T1, lower::T2)
        getdt(upper) == getdt(lower) || error("Bounds need to have equal dt")
        new(upper, lower)
    end
end
getdt(b::AsymBounds) = getdt(b.upper)
getubound(b::AsymBounds, n::Int) = getbound(b.upper, n)
getlbound(b::AsymBounds, n::Int) = -getbound(b.lower, n)
getuboundgrad(b::AsymBounds, n::Int) = getboundgrad(b.upper, n)
getlboundgrad(b::AsymBounds, n::Int) = -getboundgrad(b.lower, n)
getmaxn(b::AsymBounds) = min(getmaxn(b.upper), getmaxn(b.lower))

typealias VarAsymBounds AsymBounds{VarBound, VarBound}

typealias ConstAsymBounds AsymBounds{ConstBound, ConstBound}
ConstAsymBounds(upper::Real, lower::Real, dt::Real) =
    ConstAsymBounds(ConstBound(upper, dt), ConstBound(-lower, dt))
getubound(b::ConstAsymBounds) = getbound(b.upper)
getlbound(b::ConstAsymBounds) = -getbound(b.lower)

@compat typealias ConstBounds Union{ConstSymBounds, ConstAsymBounds}
