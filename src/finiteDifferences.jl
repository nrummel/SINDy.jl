function D0(N::Int, )
    M = N-4
    II = 1:M
    J = 3:N-2
    vals = ones(M)
    sparse(II,J,vals,M,N)
end

function D1(N::Int,h::Real)
    M = N-4
    I = zeros(Int,2*M)
    J = zeros(Int,2*M)
    val = zeros(2*M)
    # # Left endpoint
    # I[1:3] = [1,1,1]
    # J[1:3] = [1,2,3]
    # val[1:3] = 1/(2*h) * [-3, 4, -1]
    # # Right endpoint
    # I[end-2:end]= [N,N,N]
    # J[end-2:end]= [N,N-1,N-2]
    # val[end-2:end] = 1/(2*h) * [3,-4, 1]

    for i = 1:M
        startIx = (i-1)*2+1
        I[startIx:startIx+1] .= i
        J[startIx:startIx+1] = [i+1, i+3]
        val[startIx:startIx+1] = 1/(2*h) * [-1,1]
    end
    sparse(I,J,val,M,N)
end

function D2(N::Int,h::Real)
    M = N-4
    I = zeros(Int,3*M)
    J = zeros(Int,3*M)
    val = zeros(3*M)
    # Left endpoint
    # I[1:4] .= 1
    # J[1:4] = 1:4
    # val[1:4] = 1/(h^2) * [2,-5,4,-1]
    # # Right endpoint
    # I[end-3:end] .= N
    # J[end-3:end] = N:-1:N-3
    # val[end-3:end] = 1/(h^2) * [2,-5,4,-1]
    for i = 1:M
        startIx = (i-1)*3+1
        I[startIx:startIx+2] .= i
        J[startIx:startIx+2] = [i+1,i+2, i+3]
        val[startIx:startIx+2] =  1/(h^2) * [1, -2, 1]
    end
    sparse(I,J,val,M,N)
end
function D3(N::Int,h::Real)
    M = N-4
    I = zeros(Int,4*M)
    J = zeros(Int,4*M)
    val = zeros(4*M)
    # # Left endpoint
    # I[1:5] .= 1
    # I[6:10] .= 2
    # J[1:5] = 1:5
    # J[6:10] = 1:5
    # val[1:5] = 1/(h^3) * [-5/2,9,-12,7,-3/2]
    # val[6:10] = 1/(h^3) * [-5/2,9,-12,7,-3/2]
    # # Right endpoint
    # I[end-4:end] .= N
    # I[end-9:end-5] .= N-1
    # J[end-4:end] = N:-1:N-4
    # J[end-9:end-5] = N:-1:N-4
    # val[end-4:end] = -1/(h^3) * [-5/2,9,-12,7,-3/2]
    # val[end-9:end-5] = -1/(h^3) * [-5/2,9,-12,7,-3/2]
    for i = 1:M 
        # startIx = 10+4*(i-3)+1
        startIx = 4*(i-1)+1
        I[startIx:startIx+3] .= i
        J[startIx:startIx+3] = [i,i+1,i+3,i+4]
        val[startIx:startIx+3] =  1/(h^3) * [-1/2, 1, -1, 1/2]
    end
    sparse(I,J,val,M,N)
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
function D(N::Int, h::Real, order::Int)::AbstractMatrix
    if order == 0
        return D0(N)
    elseif order == 1 
        return D1(N,h)
    elseif order == 2 
        return D2(N,h)
    elseif order == 3 
        return D3(N,h)
    end 
    @error "Not implemented for order $order"
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

    DevLib  = Vector{NamedTuple}(undef, prod(alpha .+ 1))
    ix = 0
    for (i, (ai,ni,hi)) in enumerate(zip(alpha,N,h))
        for p = 0:ai
            ix += 1
            dev = D(ni,hi,p)
            DevLib[ix] = ( 
                name="D$(i==1 ? "t" : "x$(i-1)"):$p",
                dev= u -> begin # this is a bit complicated but basically make the derivative act on the correct dimension of u
                    dev*u
                    # sz = size(u)
                    # du = zeros(sz)
                    # for k = 1:sz[i]
                    #     kk = vcat(
                    #         [1:nn for nn in sz[1:i-1]], 
                    #         1:sz[k], 
                    #         [1:nn for nn in sz[i+1:end]]
                    #     )
                    #     println(kk)
                    #     @show ni,hi,p
                    #     du[kk...] = dev * u[kk...]
                    # end
                    # du
                end
            )
        end
    end
    return DevLib
end