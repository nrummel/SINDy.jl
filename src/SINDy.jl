# module SINDy
using SparseArrays, LinearAlgebra, Optim, Statistics
##
includet("example.jl")
includet("optimizers.jl")
# using FiniteDifferences
function D(N::Int,h::Real)
    I = zeros(Int,2*N+2)
    J = zeros(Int,2*N+2)
    val = zeros(2*N+2)
    # Left endpoint
    I[1:3] = [1,1,1]
    J[1:3] = [1,2,3]
    val[1:3] = 1/(2*h) * [-3, 4, -1]
    # Right endpoint
    I[end-2:end]= [N,N,N]
    J[end-2:end]= [N,N-1,N-2]
    val[end-2:end] = 1/(2*h) * [3,-4, 1]
    for i = 2:N-1
        startIx = 3+2*(i-2)+1
        I[startIx:startIx+1] .= i
        J[startIx:startIx+1] = [i-1, i+1]
        val[startIx:startIx+1] = 1/(2*h) * [-1,1]
    end
    sparse(I,J,val)
end
function D2(N::Int,h::Real)
    I = zeros(Int,3*N+2)
    J = zeros(Int,3*N+2)
    val = zeros(3*N+2)
    # Left endpoint
    I[1:4] .= 1
    J[1:4] = 1:4
    val[1:4] = 1/(h^2) * [2,-5,4,-1]
    # Right endpoint
    I[end-3:end] .= N
    J[end-3:end] = N:-1:N-3
    val[end-3:end] = 1/(h^2) * [2,-5,4,-1]
    for i = 2:N-1
        startIx = 4 + 3*(i-2) + 1
        I[startIx:startIx+2] .= i
        J[startIx:startIx+2] = [i-1,i, i+1]
        val[startIx:startIx+2] =  1/(h^2) * [1, -2, 1]
    end
    sparse(I,J,val)
end
function D3(N::Int,h::Real)
    I = zeros(Int,4*N+4)
    J = zeros(Int,4*N+4)
    val = zeros(4*N+4)
    # Left endpoint
    I[1:5] .= 1
    I[6:10] .= 2
    J[1:5] = 1:5
    J[6:10] = 1:5
    val[1:5] = 1/(h^3) * [-5/2,9,-12,7,-3/2]
    val[6:10] = 1/(h^3) * [-5/2,9,-12,7,-3/2]
    # Right endpoint
    I[end-4:end] .= N
    I[end-9:end-5] .= N-1
    J[end-4:end] = N:-1:N-4
    J[end-9:end-5] = N:-1:N-4
    val[end-4:end] = -1/(h^3) * [-5/2,9,-12,7,-3/2]
    val[end-9:end-5] = -1/(h^3) * [-5/2,9,-12,7,-3/2]
    for i = 3:N-2 
        startIx = 10+4*(i-3)+1
        I[startIx:startIx+3] .= i
        J[startIx:startIx+3] = [i-2,i-1,i+1,i+2]
        val[startIx:startIx+3] =  1/(h^3) * [-1/2, 1, -1, 1/2]
    end
    sparse(I,J,val)
end
## Get Data
tt, xx, uB, nx, nt, hx, ht = genData()
## 
FunctionLib = [
    (name="linear",fun=x -> x),
    (name="quadratic", fun=x -> x^2),
    (name="cubic", fun=x -> x^3),
    # x -> cos(x),
    # x -> sin(x),
]
TemporalDevLib = [
    (name="zero:time", dev=LinearAlgebra.I),
    (name="first:time", dev=D(nt,ht)),
    (name="second:time", dev=D2(nt,ht)),
    (name="third:time", dev=D3(nt,ht)),
]
SpatialDevLib = [
    (name="zero:space", dev=LinearAlgebra.I),
    (name="first:space", dev=D(nx,hx)),
    (name="second:space", dev=D2(nx,hx)),
    (name="third:space", dev=D3(nx,hx)),
]
## Spatial Derivatives 
N = length(FunctionLib)
Ms = length(SpatialDevLib)
Mt = length(TemporalDevLib)
K = prod(size(uB))
GG = zeros(K, N*(Ms+Mt) )
p = Progress(N*(Ms+Mt))

Ix2Name = Vector{String}(undef, N*(Ms+Mt))
# @info "Done with spatial"
for (i, (devName, dev)) in enumerate(TemporalDevLib)
    for (j, (funName, fun)) in enumerate(FunctionLib)
        next!(p; showvalues = [(:dev,devName), (:fun, funName)])
        ix = N*(i-1) + j
        Ix2Name[ix] = devName*":"*funName
        # @info "ix =$ix"
        GG[:,ix] = (dev*fun.(uB'))[:]
    end
end
for (i, (devName, dev)) in enumerate(SpatialDevLib)
    for (j, (funName, fun)) in enumerate(FunctionLib)
        next!(p; showvalues = [(:dev,devName), (:fun, funName)])
        ix = Mt*N+N*(i-1) + j
        Ix2Name[ix] = devName*":"*funName
        # @info "ix =$ix"
        GG[:,ix] = ((dev*fun.(uB))')[:]
    end
end
## Standardize GG 
shifts =  zeros(size(GG,2))
weights = zeros(size(GG,2))
for i = 1:size(GG,2)
    shifts[i] = mean(GG[:,i])
    weights[i] = std(GG[:,i])
    GG[:,i] = (GG[:,i] .- shifts[i]) / weights[i] 
end
##
IX = Mt*N+1:N*(Ms+Mt)
b = GG[:,1]
G = GG[:,IX] # only spatial derivatives
λ = .01
f = w -> norm(G*w - b) + λ * norm(w,1)
M = G'*G
function g!(grad, w)
    grad[:] = M *w + G'*b + λ * sign.(w)
end

c = 1; 
μ = .1;
wstar = [c, 0, 0, 0, μ/2, 0, 0, 0, 0]
w, pobj, times  = fista(G,b, λ,retVec=true)
w = w ./ weights[IX] * weights[1]
display([
    plot(
        scatter(y=pobj, name="Objective Value"),
        Layout(yaxis_type="log")
    
    );
    plot(
        scatter(y=diff(times), name="iteration times (s)"),
        Layout(yaxis_type="log")
    )
])
nothing

# res = optimize(f,g!,w0,LBFGS(),Optim.Options(show_trace=true,show_every=100))
# display(res)
# w = Optim.minimizer(res)

##
# uB

# end #module 

