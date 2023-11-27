using LinearAlgebra, Logging, Printf
"""
    Approximate the eigen value
    IN: 
        zkm1: estimate of the eigen vector 
        yk: A*zkm1 
        normal: flag specifying if A'A = AA'
    OUT: 
        λhat: estimate of λ
"""
function estimateλ(zkm1::AbstractVector{T}, yk::AbstractVector{T}, normal::Bool) where T 
    n = length(zkm1)
    @assert n == length(yk)
    if normal # use Raleigh-Ritz
        return  dot(zkm1,yk) / norm(zkm1)^2 
    end
    u = rand(n)
    return dot(u,yk) / dot(u, zkm1) 
end
"""
    Unaltered step for power method 
    IN: 
        A: matrix for which you are estimating eigen values for 
        zkm1: current estimate of eigen vector 
        normal: flag specifying if A'A = AA'
        _: placeholder for consistency in step functions 
    OUT: 
        λk: updated estimate for eigen value 
        zk: updated estimate for eigen vector 
        yk: A * zkm1
"""
function vanilla(A::AbstractMatrix{T}, zkm1::AbstractVector{T}, normal::Bool, ::T) where T 
    yk = A*zkm1
    zk = yk / norm(yk)
    λk = estimateλ(zkm1, yk, normal)
    return yk, zk, λk
end
"""
    Shifted step for power method 
    IN: 
        A: matrix for which you are estimating eigen values for 
        zkm1: current estimate of eigen vector 
        normal: flag specifying if A'A = AA'
        b: shi
    OUT: 
        λk: updated estimate for eigen value 
        zk: updated estimate for eigen vector 
        yk: A * zkm1
"""
function shift(A::AbstractMatrix{T}, zkm1::AbstractVector{T}, normal::Bool, b::T) where T
    yk = (A - b*I) * zkm1
    zk = yk / norm(yk)
    λk = estimateλ(zkm1, yk, normal)
    λk += b
    return yk, zk, λk
end
"""
    Shifted step for power method 
    IN: 
        A: matrix or factorization for which you are estimating eigen values for 
        zkm1: current estimate of eigen vector 
        normal: flag specifying if A'A = AA'
        _: placeholder for consistency in step functions 
    OUT: 
        λk: updated estimate for eigen value 
        zk: updated estimate for eigen vector 
        yk: A^-1 * zkm1
"""
function inv(A::Union{AbstractMatrix{T}, Factorization{T}}, zkm1::AbstractVector{T}, normal::Bool, ::T) where T 
    yk = A \ zkm1
    zk = yk / norm(yk)
    λk = estimateλ(zkm1, yk, normal)
    λk = 1 / λk
    return yk, zk, λk
end
"""
    Shifted step for power method 
    IN: 
        A: matrix or factorization for which you are estimating eigen values for 
        zkm1: current estimate of eigen vector 
        normal: flag specifying if A'A = AA'
        b: shift 
    OUT: 
        λk: updated estimate for eigen value 
        zk: updated estimate for eigen vector 
        yk: (A - b*I)^-1 * zkm1
"""
function invShift(A::Union{AbstractMatrix{T}, Factorization}, zkm1::AbstractVector{T}, normal::Bool, b::T) where T 
    if isa(A, Factorization)
        @info "  using decomp"
        yk = A \ zkm1
    else 
        @warn "  Not using Decomp (this will be inefficient)"
        yk = (A-b*I) \ zkm1
    end
    zk = yk / norm(yk)
    λk = estimateλ(zkm1, yk, normal)
    λk = 1 / λk
    λk += b
    return yk, zk, λk
end
"""
    Shifted step for power method 
    IN: 
        A: matrix for which you are estimating eigen values for 
        (optional) maxIter: maximum iterations 
        (optional) tol: error tolerance 
        (optional) b: shift 
        (optional) mode: internal step function
        (optional) updateShift: def
        (optional) ll: logging level
        (optional) retVec: flag specifying whether we 
        (optional) decomp: matrix factorization to use for inverse
    OUT: 
        λk: updated estimate for eigen value 
        zk: updated estimate for eigen vector 
        yk: (A - b*I)^-1 * zkm1
"""
function powerMethod(
    A::AbstractMatrix{T}; 
    maxIter::Int=100, 
    tol::Real=1e-12, 
    b::Real=0.0,
    mode::Function=vanilla,
    updateShift::Bool=false,
    ll::Logging.LogLevel=Logging.Warn,
    retVec::Bool=false,
    decomp::Function=lu
) where T 
    with_logger(ConsoleLogger(stderr, ll)) do
        n, m  = size(A)
        @assert n == m "A is not square"
        if updateShift 
            @assert mode==invShift "Updating shift at each iteration only implemented for inv+shifted"
        end

        # check if matrix is normal
        normal = false
        AAt = A*A'
        if norm(AAt - A'*A) / norm(AAt) < eps() 
            normal = true 
        end
        @info normal ? "A is normal using Raleigh-Ritz" : "A is not normal using standard bound (slower)"
        origA = copy(A)
        # debug info 
        if mode == vanilla 
            @info "vanilla power method"
        elseif mode == shift
            @info "shifted power method"
        elseif mode == inv
            # A = decomp(A)
            @info "inverse power method"
        elseif mode == invShift && !updateShift 
            A = decomp(A-b*I)
            @info "inverse shifted power method"
        elseif mode == invShift
            @info "modified inverse shifted power method"
        else 
            @error "$(str(mode)) not implemented"
        end

        # preallocate
        errEst = zeros(maxIter)
        ykm1 = rand(n)
        zkm1 = ykm1 / norm(ykm1)
        yk = zeros(n)
        zk = zeros(n)
        λkm1 = 1
        λk = 0

        # start iteration
        for k in 1:maxIter
            @info "k = $k"
            yk, zk, λk = mode(A, zkm1, normal, b)
            errEst[k] = abs(λk - λkm1) / abs(λkm1)
            if retVec
                errEst[k] = max( 
                    norm(origA * zk - λk * zk) / norm(zk),
                    errEst[k]
                )
            end
            errEst[k] = max(eps(), errEst[k])
            @info "  b = $b"
            @info "  λ_k = $(@sprintf "%.3f" λk)"
            @info "  error_k $(errEst[k])"
            if errEst[k] < tol 
                @info "Convergence criterion met "
                return retVec ? (λk, zk, errEst[1:k]) : (λk, errEst[1:k])
            end

            if updateShift 
                b = λk 
            end
            ykm1, zkm1, λkm1 = yk, zk, λk
        end

        @warn """Convergence criterion of $tol not met in $maxIter iterations
        error estimated at $(errEst[end])"""
        return retVec ? (λk, zk, errEst) : (λk, errEst)
    end
end