module DiffModels

import Base: rand
import Distributions: pdf, sampler, twoπ, sqrt2π
import Distributions: Sampleable, ValueSupport, Univariate, Multivariate

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
    DMBoundOutcome,
    DMBoundsOutcome,

    # methods
    rand,
    sampler,
    pdf,
    pdfu,
    pdfl,
    pdful

include("common.jl")
include("drift.jl")
include("bound.jl")
include("diffusion.jl")
include("fpt.jl")
include("sim.jl")

end # module