module DiffModels

# TODO:
# - add simulation support, ev. using the Sampler framework

include("common.jl")
include("drift.jl")
include("bound.jl")
include("diffusion.jl")
include("fpt.jl")

end # module