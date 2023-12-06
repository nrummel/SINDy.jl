module SINDy
using SparseArrays, LinearAlgebra, Statistics
using LinearAlgebra, Logging, Printf
using Convex, SCS, StaticArrays
##
include("optimizers.jl")
include("finiteDifferences.jl")

"""
    Obtain Derivatives Libraries 

    IN: 
        Lib - Library
        u - data array
    OUT: 
        G - Matrix where each column is a library function applied to the data
        b - RHS eie du/dt 
        names - vector of strings so we can identify the columns of G 
"""
function buildLinearSystem(Lib::AbstractVector{NamedTuple}, u::AbstractArray{T}) where T<:Real
    N = length(Lib)
    M = prod(size(u))
    G = Matrix{T}(undef, M, N)
    names = Vector{String}(undef, N)
    for (ix, (name, fun)) in enumerate(Lib)
        names[ix] = name
        G[:,ix] = fun(u)[:]
    end
    sz = size(u)
    pattern = length(sz) == 1 ? "Dt:1:p1" : "Dt:1:"* prod("Dx$i:0" for i = 1:length(size(u))-1)*":p1"
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
"""
    getFunctionLib: obtain a vector of function in you library 

    IN: 
        alpha - Vector or scalar of derivative order [t, x1, x2, ..., xd] 
        N - number of points in grid [Nt, Nx1, Nx2, Nx3, ..., Nx4]
        h - grid spacing [ht, hx1, hx2, hx3, ..., hx4]
        P::Int - Highest order of polynomials
        Q::Int - Highest order of trig functions
    OUT: 
        FunLib::AbstractVector{NamedTuple} - Function library combinations of 
            linear differential operators applied to non linear functions of the data
"""
function buildLib(alpha::Union{AbstractVector{Int},Int}, N::Union{AbstractVector{Int},Int}, h::Union{AbstractVector{<:Real},Real}, P::Int,Q::Int)
    DevLib = SINDy.getDevLibs(alpha, N, h)
    FunLib = SINDy.getFunctionLib(P,Q)
    Lib = Vector{NamedTuple}(undef, length(DevLib)*length(FunLib))
    ix = 0
    for (devName, dev) in DevLib
        for (funName, fun) in FunLib
            ix +=1
            Lib[ix] = (
                name=devName*funName,
                fun=uu->dev(fun(uu))
            )
        end 
    end 
    return Lib
end

end #module 

