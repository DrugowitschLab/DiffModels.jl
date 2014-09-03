# Testing:
#
# - MC estimates of mean, variance

using DiffModels
using Base.Test

function estimatemoments(d::AbstractDrift, b::AbstractBounds, n)
    t = fill(NaN, n)
    c = fill(NaN, n)
    s = sampler(d, b)
    for i = 1:n
        t[i], ci = rand(s)
        c[i] = ci ? 1.0 : 0.0
    end
    mean(t), var(t), mean(c)
end

# symmetric bounds

dmsymmeant(mu::Real, bound::Real) =
    isapprox(mu, zero(mu)) ? abs2(bound) : bound / mu * tanh(bound * mu)
function dmsymvart(mu::Real, bound::Real)
    if isapprox(mu, zero(mu))
        return 2abs2(abs2(bound)) / 3
    end
    const a = mu * bound
    bound * (tanh(a) - a * abs2(sech(a))) / mu^3
end
dmsympubound(mu::Real, bound::Real) = 1 / (1 + exp(-2mu * bound))

n = 10^5
dt = 0.001
bound = 0.9
mu = 1.1
tmu, tvar, cmu = estimatemoments(ConstDrift(mu, dt), ConstSymBounds(bound, dt), n)
@test_approx_eq_eps tmu dmsymmeant(mu, bound) 0.1
@test_approx_eq_eps tvar dmsymvart(mu, bound) 0.1
@test_approx_eq_eps cmu dmsympubound(mu, bound) 0.05
