module DiffModels

import Distributions: pdf, twoπ, sqrt2π

export
    # generic types
    AbstractDrift,
    AbstractBound,
    AbstractBounds,

    # types
    ConstDrift,
    VarDrift,
    ConstBound,
    VarBound,
    SymBounds,
    ConstBounds,
    ConstSymBounds,
    VarSymBounds,
    AsymBounds,
    ConstAsymBounds,
    VarAsymBounds,

    # methods
    pdf,
    pdfu,
    pdfl,
    pdful

# TODO:
# - add simulation support, ev. using the Sampler framework

include("common.jl")
include("drift.jl")
include("bound.jl")
include("diffusion.jl")
include("fpt.jl")

end # module