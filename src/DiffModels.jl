module DiffModels

import Distributions: pdf, twoπ, sqrt2π

export
    # types
    ConstDrift,
    VarDrift,
    ConstBound,
    VarBound,
    SymBounds,
    ConstSymBounds,
    AsymBounds,
    ConstAsymBounds,

    # methods
    pdf

# TODO:
# - add simulation support, ev. using the Sampler framework

include("common.jl")
include("drift.jl")
include("bound.jl")
include("diffusion.jl")
include("fpt.jl")

end # module