module SINDy
using SparseArrays, LinearAlgebra, Statistics
using LinearAlgebra, Logging, Printf
using Convex, SCS
##
include("optimizers.jl")
include("finiteDifferences.jl")

"""
    Obtain Derivatives Libraries 

    IN: 
        Lib - Library
        u - data array
    OUT: 
        spatial and temporal derivative libraries
"""
function buildLinearSystem(Lib, u)
    N = length(Lib)
    # M = prod(size(u).-4)
    G = nothing 
    names = Vector{String}(undef, N)
    for (ix, (name, fun)) in enumerate(Lib)
        names[ix] = name
        gi = nothing 
        # @info "ix = $ix"
        try
            gi = fun(u)[:]
            # @info "sz = $(length(gi))"
        catch 
            # @warn "here"
            gi = fun.(u)[:]
        end
        if isnothing(G)
            M = length(gi)
            G = Matrix{Real}(undef, M, N)
        end
        G[:,ix] = gi
    end
    sz = size(u)
    pattern = length(sz) == 1 ? "Dt:1:p1" : "Dt:"* prod("Dx$i:0" for i = 1:length(size(u)))*":p1"
    ix = findfirst(names .== pattern)
    y = G[:,ix]
    G = G[:,setdiff(1:N, ix)]
    names = names[setdiff(1:N, ix)];
    return G, y, names
end
"""
    getFunctionLib: obtain a vector of function in you library 

    IN: 
        P::Int - Highest order of polynomials
        Q::Int - Highest order of trig functions
    OUT: 
        FunLib::AbstractVector{NamedTuple} - Function library
"""
function getFunctionLib(P::Int,Q::Int)::AbstractVector{NamedTuple}
    @assert P >= 0 && Q >= 0
    FunLib = Vector{NamedTuple}(undef,P+2*Q) 
    for p in 1:P 
        FunLib[p] = (name="p$p",fun=u -> u .^ p)
    end
    for q in 1:Q
        ix = P+(q-1)*2+1 
        FunLib[ix:ix+1] = [(name="cos$q",fun=u->cos.(q*u)), (name="sin$q",fun=fun=u->sin.(q*u))]
    end
    return FunLib
end

function buildLib(alpha::Union{AbstractVector{Int},Int}, N::Union{AbstractVector{Int},Int}, h::Union{AbstractVector{<:Real},Real}, P::Int,Q::Int)
    DevLib = SINDy.getDevLibs(alpha, N, h)
    FunLib = SINDy.getFunctionLib(P,Q)
    Lib = Vector{NamedTuple}(undef, length(DevLib)*length(FunLib))
    ix = 0
    for (devName, dev) in DevLib
        for (funName, fun) in FunLib
            ix +=1
            Lib[ix] = (
                name=devName*":"*funName,
                fun=u->dev(fun(u))
            )
        end 
    end 
    return Lib
end


function solveWithLasso(G, y, λ)
    A = copy(G) 
    N = size(G,2)
    scales = zeros(N)
    for i in 1:N
        scales[i] = std(A[:,i])
        A[:,i] = (A[:,i] .- mean(A[:,i])) / scales[i]
    end
    b = copy(y)
    x = Convex.Variable(N)
    x.value = A'*b
    obj = minimize(norm(A*x-b) + λ*norm(x,1) )
    solve!(obj,SCS.Optimizer; silent_solver = true)
    wbias = evaluate(x) 
    nzIx = findall(abs.(wbias) .> 1e-5)
    # Could use CG or if small enough direct method... 
    x = Convex.Variable(length(nzIx))
    obj = minimize(norm(A[:,nzIx]*x-b) )
    solve!(obj,SCS.Optimizer; silent_solver = true)
    alpha = evaluate(x)
    wcvx = zeros(N)
    for (i,ix) in enumerate(nzIx)
        wcvx[ix] = alpha[i] 
    end
    wcvx ./= scales
    return wcvx
end
end #module 

