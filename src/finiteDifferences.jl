abstract type DiffMethod end
struct SparseImpl <: DiffMethod end 
struct MatFreeImpl <: DiffMethod end 

function getIx(h::Real, order::Int )
    if order == 1 
        fwd = 1/h * SA[-11/6, 3, -3/2, 1/3]
        cnt = 1/h * SA[-1/2, 0, 1/2]
    elseif order == 2 
        fwd = 1/h^2 * SA[35/12, -26/3, 19/2, -14/3, 11/12]
        cnt = 1/h^2 * SA[1, -2, 1]
    elseif order == 3 
        fwd = 1/h^3 * SA[-17/4,71/4,-59/2,49/2,-41/4,7/4]
        cnt = 1/h^3 * SA[-1/2, 1, 0, -1, 1/2]
    else 
        @error "Not implemented for order $order"
    end
    bwd = mod(order,2) == 0 ? fwd : -fwd 
    return (fwd, cnt, bwd)
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
function D(u::AbstractArray{T,N}, h::Real, order::Int, dim::Int=1; meth::DiffMethod=MatFreeImpl()) where {T<:Real,N}
    if order == 0
        return u
    end
    du = similar(u)
    fwd, cnt, bwd = getIx(h,order)
    sz = size(u)
    D = sz[dim]
    leftIx = CartesianIndices(ntuple(i->sz[i], Val(dim-1)))
    rightIx = CartesianIndices(ntuple(i->sz[dim+i], Val(N-dim)))
    if isa(meth, MatFreeImpl)
        _D_matFree!(
            du, u, 
            D, cnt, fwd, bwd,
            leftIx, rightIx
        )
    elseif isa(meth, SparseImpl)
        _D_sparse!(
            du, u, 
            D, cnt, fwd, bwd,
            leftIx, rightIx
        )
    else 
        throw("Not Implemented")
    end
    return du
end

function _D_matFree!(
    du::AbstractArray{T,N}, u::AbstractArray{T,N}, D::Int,
    cnt::SVector{K,T}, fwd::SVector{J,T}, bwd::SVector{J,T},
    leftIx::CartesianIndices, rightIx::CartesianIndices
) where {T<:Real,N,J,K}
    bnd = Int(floor(K/2))
    @assert all(size(du) .== size(u)) "Input and output must be of the same size"

    @inbounds for l in leftIx,r in rightIx
        for i in 1:bnd 
            ix = SVector(ntuple(x->i-1+x, Val(J)))
            du[l,i,r] = dot(fwd, @view u[l,ix,r])
        end
        for i = bnd+1:D-bnd-1 
            ix = SVector(ntuple(x->i-bnd+x-1, Val(K)))
            du[l,i,r] = dot(cnt, @view u[l,ix,r])
        end
        for i = D-bnd:D 
            ix = SVector(ntuple(x->i-x+1, Val(J)))
            du[l,i,r] =dot(bwd, @view u[l,ix,r])
        end
    end
    nothing
end
function _D_sparse!(
    du::AbstractArray{T,N}, u::AbstractArray{T,N}, D::Int,
    cnt::SVector{K,T}, fwd::SVector{J,T}, bwd::SVector{J,T},
    leftIx::CartesianIndices, rightIx::CartesianIndices
) where {T<:Real,N,J,K}
    # pre allocating here would help with speed
    rowIx = Int[]
    colIx = Int[]
    nzVal = Float64[]
    bnd = Int(floor(length(cnt)/2))
    for i = 1:D
        if i <= bnd
            k = length(fwd)
            append!(rowIx, i*ones(k))
            append!(colIx, i:i+k-1)
            append!(nzVal, fwd)
        elseif D-bnd <= i 
            k = length(bwd)
            append!(rowIx, i*ones(k))
            append!(colIx, i:-1:i-k+1)
            append!(nzVal, bwd)
        else 
            k = length(cnt)
            k2 = Int(floor(k/2))
            append!(rowIx, i*ones(k))
            append!(colIx, i-k2:i+k2)
            append!(nzVal, cnt)
        end
    end
    ## build the spars matrix 
    DD = sparse(rowIx,colIx,nzVal,D,D)
    ## Apply it 
    for l in leftIx, r in rightIx
        du[l,:,r] = DD * @view u[l,:,r]
    end 
end
"""
    Wrapper to help when we pass non arrays
"""
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