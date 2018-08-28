
# Returns the gradient of x, computed by finited differences
#
# Except for the first and last element, all gradients are computed by central
# finite differences
function fdgrad(x::AbstractVector{T}, dt::Real) where T
    n = length(x)
    GType = typeof((zero(T) - zero(T)) / dt)
    g = Array{GType}(undef, n)
    if n == 1
        g[1] = zero(GType)
    else
        g[1] = (x[2] - x[1]) / dt
        for i = 2:(n-1)
            g[i] = (x[i+1] - x[i-1]) / 2dt
        end
        g[n] = (x[n] - x[n-1]) / dt
    end
    return g
end
