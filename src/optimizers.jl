includet("powerMethod.jl")
function soft_thresh(x::AbstractVector, λ::Real)
    return sign.(x) .* max.(abs.(x) .- λ, 0.)
end
function fista(A::AbstractMatrix, b::AbstractVector, λ::Real; maxit::Int=10000, errThresh::Real=1e-8,retVec::Bool=false)
    x = zeros(size(A,2))
    t = 1
    z = copy(x)
    L = size(A,1) < 1e6 ? norm(A)^2 : powerMethod(A*A', tol=1e-8) 

    times = zeros(maxit)
    pobj = zeros(maxit)
    time0 = time_ns()
    for i in 1:maxit
        xold = copy(x)
        z = z + A'*(b - A*z) / L
        x = soft_thresh(z, λ / L)
        t0 = t
        t = (1. + sqrt(1. + 4. * t ^ 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        pobj[i] = 1/2. *norm(A*x -b)^2 + λ * norm(x,1)
        times[i] = (time_ns() - time0) / 1e9
        if norm(x - xold)/norm(x) < errThresh 
            pobj = pobj[1:i]
            times = times[1:i]
            return retVec ? (x, pobj, times) : x
        end
    end
    
    @warn "Max Iterations Reached, pobj= $(pobj[end])"
    return retVec ? (x, pobj, times) : x
end