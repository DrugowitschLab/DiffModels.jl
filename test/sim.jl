# Testing:
#
# - MC estimates of mean, variance
#
# TODO: add variance testing for asymmetric bounds

using DiffModels
using Base.Test

nanmean(x) = mean(x[!isnan(x)])
nanvar(x) = var(x[!isnan(x)])

function estimatemoments(d::AbstractDrift, b::AbstractBounds, n)
    t = fill(NaN, n)
    c = fill(NaN, n)
    s = sampler(d, b)
    for i = 1:n
        t[i], ci = rand(s)
        c[i] = ci ? 1.0 : 0.0
    end
    [nanmean(t), nanvar(t), mean(c),
     nanmean(t[c .== 1.0]), nanvar(t[c .== 1.0]),
     nanmean(t[c .== 0.0]), nanvar(t[c .== 0.0])]
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

# normal/exponential sampler for small mu
bound = 0.9
mu = 0.5
m = estimatemoments(ConstDrift(mu, dt), ConstSymBounds(bound, dt), n)
@test_approx_eq_eps m[1] dmsymmeant(mu, bound) 0.1
@test_approx_eq_eps m[2] dmsymvart(mu, bound) 0.1
@test_approx_eq_eps m[3] dmsympubound(mu, bound) 0.05

# inverse-normal sampler for large mu
bound = 0.9
mu = 2
m = estimatemoments(ConstDrift(mu, dt), ConstSymBounds(bound, dt), n)
@test_approx_eq_eps m[1] dmsymmeant(mu, bound) 0.1
@test_approx_eq_eps m[2] dmsymvart(mu, bound) 0.1
@test_approx_eq_eps m[3] dmsympubound(mu, bound) 0.05

# generic sampler
bound = 0.9
mu = 1.1
m = estimatemoments(VarDrift([mu], dt, 15.0), ConstSymBounds(bound, dt), n)
@test_approx_eq_eps m[1] dmsymmeant(mu, bound) 0.1
@test_approx_eq_eps m[2] dmsymvart(mu, bound) 0.1
@test_approx_eq_eps m[3] dmsympubound(mu, bound) 0.05

# asymmetric bounds

dmasymmeantup(mu::Real, blo::Real, bup::Real) = isapprox(mu, zero(mu)) ? 
    (abs2(bup) - 2bup * blo) / 3 :
    (bup - blo) / mu * coth((bup - blo) * mu) + blo / mu * coth(- blo * mu)
dmasymmeantlo(mu::Real, blo::Real, bup::Real) = isapprox(mu, zero(mu)) ?
    (abs2(blo) - 2bup * blo) / 3 :
    (bup - blo) / mu * coth((bup - blo) * mu) - bup / mu * coth(bup * mu)
dmasympubound(mu::Real, blo::Real, bup::Real) = isapprox(mu, zero(mu)) ?
    blo / (blo + bup) : (exp(-2mu * blo) - 1) / (exp(-2mu * blo) - exp(-2mu * bup))

# fast sampler for constant drift/bounds
blo = -1.5
bup = 1.0
mu = 1.1
m = estimatemoments(ConstDrift(mu, dt), ConstAsymBounds(bup, blo, dt), n)
@test_approx_eq_eps m[3] dmasympubound(mu, blo, bup) 0.05
@test_approx_eq_eps m[4] dmasymmeantup(mu, blo, bup) 0.1
@test_approx_eq_eps m[6] dmasymmeantlo(mu, blo, bup) 0.1

# generic sampler
blo = -1.5
bup = 1.0
mu = 1.1
m = estimatemoments(VarDrift([mu], dt, 15.0), ConstAsymBounds(bup, blo, dt), n)
@test_approx_eq_eps m[3] dmasympubound(mu, blo, bup) 0.05
@test_approx_eq_eps m[4] dmasymmeantup(mu, blo, bup) 0.1
@test_approx_eq_eps m[6] dmasymmeantlo(mu, blo, bup) 0.1
