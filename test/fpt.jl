# Testing:
#
# - consistency of first-passage time densities

using DiffModels
using Base.Test

function testpdfequal(d::AbstractDrift, b::AbstractBounds, tmax,
                      g1ref::Vector{Float64}, g2ref::Vector{Float64}, tol)
    g1, g2 = pdf(d, b, tmax)
    @test length(g1) == length(g2)
    @test length(g1) == length(g1ref)
    @test length(g2) == length(g2ref)
    for n = 1:length(g1)
        @test_approx_eq_eps g1[n] g1ref[n] tol
        @test_approx_eq_eps g2[n] g2ref[n] tol
    end
end

# symmetric bounds

mu = 1.2
bound = 1.1
dt = 0.005
tmax = 3.0
maxn = length(0:dt:tmax)

g1, g2 = pdf(VarDrift(fill(mu, maxn), dt), 
             VarAsymBounds(VarBound(fill(bound, maxn), dt),
                           VarBound(fill(bound, maxn), dt)), tmax)
testpdfequal(ConstDrift(mu, dt), ConstSymBounds(bound, dt), tmax, g1, g2, 1e-4)
testpdfequal(ConstDrift(mu, dt), ConstAsymBounds(bound, -bound, dt), tmax, g1, g2, 1e-4)

# asymmetric bounds

mu = 1.2
bounds = [1.1, 0.5]
dt = 0.005
tmax = 3.0
maxn = length(0:dt:tmax)

g1, g2 = pdf(VarDrift(fill(mu, maxn), dt), 
             VarAsymBounds(VarBound(fill(bounds[1], maxn), dt),
                           VarBound(fill(bounds[2], maxn), dt)), tmax)
testpdfequal(ConstDrift(mu, dt), ConstAsymBounds(bounds[1], -bounds[2], dt), tmax, g1, g2, 1e-4)
