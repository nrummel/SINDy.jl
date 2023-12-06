include("powerMethod.jl")
abstract type StoppingCriterion end
struct RelErr<:StoppingCriterion end
struct AbsErr<:StoppingCriterion end
struct GradNorm<:StoppingCriterion end
struct AnyOrAll<:StoppingCriterion end
struct FcnHist 
    k::Int
    x::AbstractVector 
    iterTime::Real
    absErr::Real
    relErr::Real
    gradNorm::Union{Nothing, Real}
    obj::Real
    function FcnHist(
        k::Int,
        iterTime::Real,
        xk::AbstractVector,
        absErr::Real,
        gkp1::AbstractVector, 
        f::Function,
        prox_obj::Function
    )
        obj = f(xk) + prox_obj(xk)
        return new(k, xk, iterTime, absErr, absErr/norm(xk), norm(gkp1), obj)
    end
end
function fval(fcnHist::FcnHist, crit::RelErr)
    return fcnHist.relErr
end
function fval(fcnHist::FcnHist, crit::AbsErr)
    return fcnHist.absErr
end
function fval(fcnHist::FcnHist, crit::GradNorm)
    return fcnHist.gradNorm
end
function fval(fcnHist::FcnHist, crit::AnyOrAll)
    return min(fcnHist.relErr,fcnHist.absErr,fcnHist.gradNorm)
end
function summary(fcnHist::FcnHist, crit::StoppingCriterion, tol::Real)
    """
        Summary of Gradient Descent:
        stopping criterion = $(typeof(crit))
        tolerance          = $(tol)
        k                  = $(fcnHist.k)
        iterTime           = $(fcnHist.iterTime)
        |xk - xkm1| / |xk| = $(fcnHist.relErr)
        |xk - xkm1|        = $(fcnHist.relErr)
        |gk|               = $(fcnHist.gradNorm)
        f(xk)              = $(fcnHist.obj)
    """
end
INIT_DEBUG_STR = "|\tk\t|\titerTime\t|\tlog(relErr)\t|\tlog(absErr)\t|\tlog(gradNorm)\t|"
function debugStr(fcnHist::FcnHist)
    @sprintf "|\t%d\t|\t%1.8f\t|\t%1.4f\t\t|\t%1.4f\t\t|\t%1.4f\t\t|" fcnHist.k  norm(fcnHist.iterTime, Inf) log10(fcnHist.relErr) log10(fcnHist.absErr) log10(fcnHist.gradNorm)
end
abstract type StepMethod end
struct ConstantStep<:StepMethod
    L::Real
end
function step(method::ConstantStep, x::AbstractVector, p::AbstractVector, prox::Function)
    return prox(x + method.L * p, method.L)
end 
# example for least squares this is p->dot(p,p)/ dot(p, A*p)
struct CustomStep<:StepMethod
    stepFun::Function
end
function step(method::CustomStep, x::AbstractVector, p::AbstractVector, prox::Function)
    return prox(method.stepFun(x, p), 0)
end 
struct ArmilloStep<:StepMethod
end
function step(method::ArmilloStep, x::AbstractVector, p::AbstractVector)
    @error "Not Implemented yet"
end 
"""
    Fixed point iteration method 'steepest decent' for root finding to the linear system of equations f(x)=A*x-b=0
"""
function gradientDescent(
    f::Function, 
    grad::Function,
    x0::AbstractVector{<:Real}; 
    prox::Function=(x,a)->x,
    prox_obj::Function=x->0,
    accel::Bool=false,
    restart::Int=0,
    stepMethod::StepMethod=ConstantStep(0.01),
    crit::StoppingCriterion=AnyOrAll(),
    tol::Real=1.0e-8, 
    maxIter::Int=10000,
    debugFreq::Int= 0,
    ll::Logging.LogLevel=Logging.Warn
)
    with_logger(ConsoleLogger(stderr, ll)) do 
        xk = x0
        kk = 0
        iterTime = 0
        fcnHist = Array{FcnHist}(undef,maxIter)
        absErr = NaN
        debugFreq > 0 && (@info INIT_DEBUG_STR)
        for k = 1:maxIter
            # println("k=$k")
            start = time_ns()
            xkm1 = xk
            gk = -grad(xk)
            fcnHist[k] = FcnHist(k-1, iterTime, xk, absErr, gk, f, prox_obj)
            # print debug information periodically if desired
            (debugFreq > 0 && mod(k-1, debugFreq) == 0 ) && (@info debugStr(fcnHist[k]))
            if fval(fcnHist[k],crit) < tol 
                @info summary(fcnHist[k], crit, tol)
                return xk, fcnHist[1:k]
            end
            z = step(stepMethod, xkm1, gk, prox)
            # z = xkm1 + stepMethod.L * gk
            absErr = norm(z - xkm1) 
            if accel 
                kk += 1
                if restart < 0 && kk > -restart && absErr > mean([fh.absErr for fh in fcnHist[k+restart+1:k]])
                    kk = 0 
                end
                xk = z + kk/(kk+3)*(z - xkm1) # Neystrov step
                absErr = norm(xk - xkm1) 
            else  
                xk = z
            end
            iterTime = (time_ns() - start)*1e-9
        end
        @warn "SD: Max iter ($maxIter) reached without convergence, $(fval(fcnHist[end],crit)) < tol = $tol"
        return xk, fcnHist
    end
end

function soft_thresh(x::AbstractVector, λ::Real)
    return sign.(x) .* max.(abs.(x) .- λ, 0.)
end
function lassoSolver(
    A::AbstractMatrix, 
    b::AbstractVector, 
    λ::Real; 
    L::Union{Nothing, Real}=nothing,
    x0::Union{Nothing, AbstractVector}=nothing,
    kwargs...)

    f(x) = 1/2* norm(A*x-b)^2
    grad(x) = A'*( A*x - b )
    n = size(A,1)
    if isnothing(L)  
        if n < 1e6 
            L = norm(A)^2
        else
            L = 1.2*powerMethod(A, At=A',x=x0, iters=10, tol=1e-3)^2
        end
    end

    isnothing(x0) && (x0 = A'*b) 
    prox_fcn(x::AbstractVector) = λ*norm(x,1) 
    
    return gradientDescent(f,grad,x0;stepMethod=ConstantStep(1/L),
                    prox=soft_thresh, prox_obj=prox_fcn,
                    accel=true, 
                    kwargs...)
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
    nzIx = findall(abs.(wbias) .> 10.0^-λ)
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
##
function MatchingPursuit(G,y, maxNz = 3 )
    N= size(G,2)
    IX = Int[]
    beta_star = nothing 
    for k = 1:maxNz
        left_over_ix = setdiff(1:N, IX)
        T = eltype(y)
        beta = Vector{Vector{T}}(undef, length(left_over_ix))
        scores = zeros(length(left_over_ix))
        for (i,n) = enumerate(left_over_ix)
            this_ix = vcat(IX, [n])
            beta[i] = ((G[:,this_ix]'*G[:,this_ix]) \ (G[:,this_ix]' * y ))
            scores[i] = norm(G[:,this_ix] * beta[i] - y)
        end 
        val, ix = findmin(abs.(scores))
        push!(IX, left_over_ix[ix])
        beta_star = zeros(N)
        beta_star[IX] = beta[ix]
        # @info """
        #     Number of Nz $k: 
        #         Obj Val     : $(scores[ix])
        #         Coeff Names : $(names_approx[IX])
        #         β           : $(round.(beta_star',sigdigits=3))
        # """
    end
    return beta_star
end
##
function OrthogonnalMatchingPursuit(G,y, maxNz = 3 )
    N= size(G,2)
    IX = Int[]
    beta_star = nothing 
    for k = 1:maxNz
        left_over_ix = setdiff(1:N, IX)
        T = eltype(y)
        beta = Vector{Vector{T}}(undef, length(left_over_ix))
        scores = zeros(length(left_over_ix))
        for (i,n) = enumerate(left_over_ix)
            this_ix = vcat(IX, [n])
            beta[i] = ((G[:,this_ix]'*G[:,this_ix]) \ (G[:,this_ix]' * y ))
            scores[i] = norm(G[:,this_ix] * beta[i] - y)
        end 
        val, ix = findmin(abs.(scores))
        push!(IX, left_over_ix[ix])
        beta_star = zeros(N)
        beta_star[IX] = beta[ix]
        # @info """
        #     Number of Nz $k: 
        #         Obj Val     : $(scores[ix])
        #         Coeff Names : $(names_approx[IX])
        #         β           : $(round.(beta_star',sigdigits=3))
        # """
    end
    return beta_star
end