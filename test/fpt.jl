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
        @test g1[n] ≈ g1ref[n] atol=tol
        @test g2[n] ≈ g2ref[n] atol=tol
    end
end

# symmetric bounds

mu = 1.2
bound = 1.1
t = 0.5
dt = 0.005
tmax = 3.0
maxn = length(0:dt:tmax)

@test (pdful(ConstDrift(mu, dt), ConstSymBounds(bound, dt), t)[1] ≈ 
       pdfu(ConstDrift(mu, dt), ConstSymBounds(bound, dt), t))
@test (pdful(ConstDrift(mu, dt), ConstSymBounds(bound, dt), t)[2] ≈
       pdfl(ConstDrift(mu, dt), ConstSymBounds(bound, dt), t))
@test (pdful(ConstDrift(mu, dt), ConstAsymBounds(bound, -bound, dt), t)[1] ≈
       pdfu(ConstDrift(mu, dt), ConstAsymBounds(bound, -bound, dt), t))
@test (pdful(ConstDrift(mu, dt), ConstAsymBounds(bound, -bound, dt), t)[2] ≈
       pdfl(ConstDrift(mu, dt), ConstAsymBounds(bound, -bound, dt), t))

g1, g2 = pdf(VarDrift(fill(mu, maxn), dt), 
             VarAsymBounds(VarBound(fill(bound, maxn), dt),
                           VarBound(fill(bound, maxn), dt)), tmax)
testpdfequal(ConstDrift(mu, dt), ConstSymBounds(bound, dt), tmax, g1, g2, 1e-4)
testpdfequal(ConstDrift(mu, dt), ConstAsymBounds(bound, -bound, dt), tmax, g1, g2, 1e-4)
testpdfequal(ConstDrift(mu, dt), LinearSymBounds(bound, 0.0, dt), tmax, g1, g2, 1e-4)
testpdfequal(ConstDrift(mu, dt),
             AsymBounds{LinearBound, LinearBound}(LinearBound(bound, 0.0, dt),
                                                  LinearBound(bound, 0.0, dt)),
             tmax, g1, g2, 1e-4)

# asymmetric bounds

mu = 1.2
bounds = [1.1, 0.5]
bound_slopes = [-0.02, -0.01]
t = 0.5
dt = 0.005
tmax = 3.0
maxn = length(0:dt:tmax)

@test (pdful(ConstDrift(mu, dt), ConstAsymBounds(bounds[1], -bounds[2], dt), t)[1] ≈
       pdfu(ConstDrift(mu, dt), ConstAsymBounds(bounds[1], -bounds[2], dt), t))
@test (pdful(ConstDrift(mu, dt), ConstAsymBounds(bounds[1], -bounds[2], dt), t)[2] ≈
       pdfl(ConstDrift(mu, dt), ConstAsymBounds(bounds[1], -bounds[2], dt), t))

g1, g2 = pdf(VarDrift(fill(mu, maxn), dt), 
             VarAsymBounds(VarBound(fill(bounds[1], maxn), dt),
                           VarBound(fill(bounds[2], maxn), dt)), tmax)
testpdfequal(ConstDrift(mu, dt), ConstAsymBounds(bounds[1], -bounds[2], dt), tmax, g1, g2, 1e-4)
testpdfequal(ConstDrift(mu, dt),
             AsymBounds{LinearBound, LinearBound}(LinearBound(bounds[1], 0.0, dt),
                                                  LinearBound(bounds[2], 0.0, dt)),
             tmax, g1, g2, 1e-4)

g1, g2 = pdf(VarDrift(fill(mu, maxn), dt),
             VarAsymBounds(VarBound(bounds[1] + bound_slopes[1] * dt * (0:maxn-1), dt), 
                           VarBound(bounds[2] + bound_slopes[2] * dt * (0:maxn-1), dt)),
             tmax)
testpdfequal(ConstDrift(mu, dt),
             AsymBounds{LinearBound, LinearBound}(LinearBound(bounds[1], bound_slopes[1], dt),
                                                  LinearBound(bounds[2], bound_slopes[2], dt)),
             tmax, g1, g2, 1e-4)
