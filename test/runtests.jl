using SINDy
using Test, PlotlyJS
using Convex, ECOS
## Test the finite difference methods
# f = sin 
# DD = D3
# df =x-> -1*cos(x)
function testFiniteDiffMethod(f::Function, df::Function, DD::Function; debug::Bool=false)
    h = 0.1
    xx = collect(-10:h:10)
    N = length(xx)
    yy = DD(N, h) * f.(xx) 
    err = max.(( yy - df.(xx)), eps())
    if debug
        display([
            plot(
            [
                scatter(x=xx,y=yy, name="approx"),
                scatter(x=xx,y=df.(xx), name="exact"),
            ]
            );
            plot(
                [
                    scatter(x=xx, y= h^2*ones(N), marker_color="black", line_dash="dash", name="h^2"),
                    scatter(x=xx, y= err, name="rel err")
                ],
                Layout(
                    yaxis=attr(
                        type="log"
                    )
                )
            )
        ])
    end
    return maximum(err) < 10*h
end

function testFISTA()
    n = 1000
    tt = range(0,1,n)
    wstar = [1,2,3,0,0]
    m = length(wstar)
    A = zeros(n,m)
    for k = 1:m 
        A[:,k] = tt.^(k-1)
    end
    y = A*wstar
    A += 1e-3*rand(n,m)
    F(x) = 1/2*norm(A*x-y)^2 + λ*norm(x,1)

    # Normalize A  
    G = zeros(size(A))
    scales = zeros(m)
    for k = 1:m 
        scales[k] = std(A[:,k])
        G[:,k] = (A[:,k] .- mean(A[:,k])) / scales[k]
    end
    b = (y .- mean(y)) / std(y)
    λ = 2
    tol=1e-8
    debugFreq= 50
    ll=Logging.Info
    w, fcnHist  = lassoSolver(G,b, λ;tol=tol, debugFreq=debugFreq,ll=ll, crit=GradNorm(), maxIter=Int(1e4), restart=-2);
    # Correct for normalization 
    

    x     = Variable(m)
    obj   = minimize(sumsquares(G*x-b)/2 + λ*norm(x,1) )
    solve!(obj,ECOS.Optimizer)
    w = evaluate(x)
    (w ./ scales ) * std(y)
    @info F(w)
    @info F(wstar)
    #highPrecision = {'solver':cvx.ECOS,'max_iters':400,'abstol':1e-13,'reltol':1e-13}
    # highPrecision = {'solver':cvx.ECOS,'max_iters':5000,'abstol':1e-16,'reltol':1e-16,'feastol':1e-16,'verbose':False}

    return norm(w - wstar)/norm(wstar) < 10 * errThresh
end

function testGradDescent()
    n = 10
    m = n
    A = rand(n,m)
    A = A*A' + I
    b = rand(n)
    tol
    debugFreq= 10
    ll=Logging.Info
    f(x) = 1/2 * norm(A*x - b)^2
    grad(x) = A'*(A*x -b)
    function stepFun(x::AbstractVector,::AbstractVector) 
        p = b - A*x
        return x + dot(p,p)/dot(p, A*p)*p
    end
    x0 = zeros(m) 
    xstar = A \ b 
    x, _ = gradientDescent(f,grad,x0; stepMethod=CustomStep(stepFun), debugFreq=debugFreq,ll=ll,tol=tol);
    return norm(x-xstar) / norm(xstar) < tol * 10
end
@testset "SINDy.jl" begin
    @test testFiniteDiffMethod(sin, cos, SINDy.D)
    @test testFiniteDiffMethod(sin, x->-sin(x), SINDy.D2)
    @test testFiniteDiffMethod(sin, x->-cos(x), SINDy.D3)
    @test testFISTA()
end
