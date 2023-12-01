function D0(N::Int, ::Real)
    D0(N)
end
function D0(N::Int)
    M = N
    II = 1:M
    J = 1:M
    vals = ones(M)
    sparse(II,J,vals,N,N)#M,N)
end
function D1(N::Int,h::Real)
    M = N-2
    K = 2*M+6
    I = zeros(Int,K)
    J = zeros(Int,K)
    val = zeros(K)
    # # Left endpoint
    I[1:3] = [1,1,1]
    J[1:3] = [1,2,3]
    val[1:3] = 1/(2*h) * [-3, 4, -1]
    # Right endpoint
    I[end-2:end]= [N,N,N]
    J[end-2:end]= [N,N-1,N-2]
    val[end-2:end] = 1/(2*h) * [3,-4, 1]
    for i = 1:M
        startIx = (i-1)*2+4
        ii = i+1
        I[startIx:startIx+1] .= ii
        J[startIx:startIx+1] = [ii-1, ii+1]#[i+1, i+3]
        val[startIx:startIx+1] = 1/(2*h) * [-1,1]
    end
    sparse(I,J,val,N,N)#M,N)
end

function D2(N::Int,h::Real)
    M = N-2
    K = 3*M+8
    I = zeros(Int,K)
    J = zeros(Int,K)
    val = zeros(K)
    # Left endpoint
    I[1:4] .= 1
    J[1:4] = 1:4
    val[1:4] = 1/(h^2) * [2,-5,4,-1]
    # Right endpoint
    I[end-3:end] .= N
    J[end-3:end] = N:-1:N-3
    val[end-3:end] = 1/(h^2) * [2,-5,4,-1]
    for i = 1:M
        startIx = (i-1)*3+5
        ii = i+1
        I[startIx:startIx+2] .= ii
        J[startIx:startIx+2] = [ii-1,ii,ii+1]#[i+1,i+2, i+3]
        val[startIx:startIx+2] =  1/(h^2) * [1, -2, 1]
    end
    sparse(I,J,val,N,N)#M,N)
end
"""
    Approximate Derivatives with finite difference methods

    IN: 
        N::Int     - number of data points
        h::Real    - step size 
        order::Int - highest order of derivatives
    OUT: 
        Operator that that acts on N x M matrices
"""
function D(N::Int, h::Real, order::Int)
    if order == 0
        return LinearAlgebra.I
    elseif order == 1 
        fwd =  1/ h * [-11/6, 3, -3/2, 1/3]	# 3rd order
        # fwd = 1/ h * [3/2,-2, 1/2] # 2nd order
        cnt = 1 / h * [-1/2, 0, 1/2]
    elseif order == 2 
        fwd = 1/h^2 * [35/12, -26/3, 19/2, -14/3, 11/12] # 3rd order
        # fwd = 1/h^2 * [2,-5,4,-1] # 2nd order
        cnt = 1/h^2 * [1, -2, 1]
    elseif order == 3 
        fwd = 1/h^3 *[-17/4,71/4,-59/2,49/2,-41/4,7/4] # 3rd order
        # fwd = 1/h^3 * [-5/2,9,-12,7,-3/2] # 2nd order  
        cnt = 1/h^3 * [-1/2, 1, 0, -1, 1/2]
    else 
        @error "Not implemented for order $order"
    end
    bwd = mod(order,2) == 0 ? fwd : -fwd
 
    I = Int[]
    J = Int[]
    val = Float64[]
    bnd = Int(floor(length(cnt)/2))
    for i = 1:N
        if i <= bnd
            k = length(fwd)
            I = vcat(I, i*ones(k))
            J = vcat(J, i:i+k-1)
            val = vcat(val, fwd)
        elseif N-bnd+1 <= i
            k = length(bwd)
            I = vcat(I, i*ones(k))
            J = vcat(J, i:-1:i-k+1)
            val = vcat(val, bwd)
        else 
            k = length(cnt)
            k2 = Int(floor(k/2))
            I = vcat(I, i*ones(k))
            J = vcat(J, i-k2:i+k2)
            val = vcat(val, cnt)
        end
    end
    sparse(I,J,val,N,N)
end

function fwdIx(ix::CartesianIndex, fwd::AbstractVector, dim::Int) 
    J = length(fwd)
    k = ix[dim]
    IX = Vector{typeof(ix)}(undef, J)
    for (j,kk) in enumerate(k:k+J-1)
        tmp = Any[Tuple(ix)...]
        tmp[dim] = kk 
        IX[j] = CartesianIndex(Tuple(tmp))
    end
    IX
end
function bwdIx(ix::CartesianIndex, bwd::AbstractVector, dim::Int) 
    J = length(bwd)
    k = ix[dim]
    IX = Vector{typeof(ix)}(undef, J)
    for (j,kk) in enumerate(k:-1:k-J+1)
        tmp = Any[Tuple(ix)...]
        tmp[dim] = kk 
        IX[j] = CartesianIndex(Tuple(tmp))
    end
    IX
end
function cntIx(ix::CartesianIndex, cnt::AbstractVector, dim::Int) 
    J = length(cnt)
    k = ix[dim]
    k2 =  Int(floor(J/2))
    IX = Vector{typeof(ix)}(undef, J)
    for (j,kk) in enumerate(k-k2:k+k2)
        tmp = Any[Tuple(ix)...]
        tmp[dim] = kk 
        IX[j] = CartesianIndex(Tuple(tmp))
    end
    IX
end



function fwdIx(fwd::AbstractVector, dim::Int, N::Int) 
    J = length(fwd) - 1
    Rpre = [0:0  for _ in 1:dim-1]
    Rpost = [0:0  for _ in dim+1:N]
    CartesianIndices(Tuple([Rpre...,0:J, Rpost...]))
end
function bwdIx(bwd::AbstractVector, dim::Int,N::Int) 
    J = length(bwd) - 1
    Rpre = [0:0  for _ in 1:dim-1]
    Rpost = [0:0  for _ in dim+1:N]
    CartesianIndices(Tuple([Rpre...,0:-1:-J,Rpost...]))
end
function cntIx(cnt::AbstractVector, dim::Int,N::Int) 
    J = Int(floor(length(cnt)/2)) 
    Rpre = [0:0  for _ in 1:dim-1]
    Rpost = [0:0  for _ in dim+1:N]
    CartesianIndices(Tuple([Rpre...,-J:J,Rpost...]))
end
"""
    Approximate Derivatives with finite difference methods

    IN: 
        u::        - data
        h::Real    - step size 
        order::Int - highest order of derivatives
        dim::Int   - dimention of the derivative
    OUT: 
        Operator that that acts on N x M matrices
"""
function D(u::AbstractArray{T}, h::Real, order::Real, dim::Int=1) where T<:Real
    du = similar(u)
    _D!(du,u,h,order,dim)
    du 
end
function _D!(du::AbstractArray{T}, u::AbstractArray{T}, h::Real, order::Real, dim::Int) where T<:Real
    if order == 0
        du[:] .= u[:]
        return
    elseif order == 1 
        fwd =  1/ h * [-11/6, 3, -3/2, 1/3]	# 3rd order
        # fwd = 1/ h * [3/2,-2, 1/2] # 2nd order
        cnt = 1 / h * [-1/2, 0, 1/2]
    elseif order == 2 
        fwd = 1/h^2 * [35/12, -26/3, 19/2, -14/3, 11/12] # 3rd order
        # fwd = 1/h^2 * [2,-5,4,-1] # 2nd order
        cnt = 1/h^2 * [1, -2, 1]
    elseif order == 3 
        fwd = 1/h^3 *[-17/4,71/4,-59/2,49/2,-41/4,7/4] # 3rd order
        # fwd = 1/h^3 * [-5/2,9,-12,7,-3/2] # 2nd order  
        cnt = 1/h^3 * [-1/2, 1, 0, -1, 1/2]
    else 
        @error "Not implemented for order $order"
    end

    sz = size(u)
    N = length(sz)
    @assert all(size(du) .== sz)

    bwd = mod(order,2) == 0 ? fwd : -fwd
    fIx = fwdIx(fwd,dim,N)
    bIx = bwdIx(bwd,dim,N)
    cIx = cntIx(cnt,dim,N)
    
    bnd = Int(floor(length(cnt)/2))
    for ix = CartesianIndices(u)
        if ix[dim]  <= bnd
            du[ix] = dot(u[ix .+ fIx], fwd)
        elseif sz[dim]-bnd <= ix[dim] 
            du[ix] = dot(u[ix .+ bIx], bwd)
        else 
            du[ix] = dot(u[ix .+ cIx], cnt)
        end
    end
    nothing
end
function getDevLibs(alpha::Int, N::Int, h::Real)
    getDevLibs([alpha], [N],[h])
end
"""
    Obtain Derivatives Libraries 

    IN: 
        alpha - Vector or scalar of derivative order [t, x1, x2, ..., xd] 
        N - number of points in grid [Nt, Nx1, Nx2, Nx3, ..., Nx4]
        h - grid spacing [ht, hx1, hx2, hx3, ..., hx4]
    OUT: 
        derivative library
"""
function getDevLibs(alpha::AbstractVector{Int}, N::AbstractVector{Int}, h::AbstractVector{<:Real})
    @assert length(alpha) == length(N) == length(h)
    IX = CartesianIndices(Tuple([0:ai for ai = alpha]))
    DevLib = Vector{NamedTuple}(undef, length(IX))
    for (i,ix) in enumerate(IX)
        name = ""
        dev = Vector{Function}(undef, length(ix))
        for (dim,ai) = enumerate(Tuple(ix))
            name *= dim==1 ? "Dt:$ai:" : "Dx$(dim-1):$ai:"
            if dim ==1 
                dev[dim] =u -> D(u, h[dim], ai, dim)
            else 
                dev[dim] =u -> dev[dim-1](D(u, h[dim], ai, dim))
            end
        end
        # dim = 1 
        # ai = alpha[dim]
        # dev = u -> D(u, h[dim], ai, dim)
        DevLib[i] = ( 
                name=name,
                dev=dev[end]
            )
    end
    return DevLib
end