using SINDy,Test
using Convex, SCS, PlotlyJS
using Statistics, LinearAlgebra, Logging
## Test the finite difference methods
# f = sin 
# DD = D3
# df =x-> -1*cos(x)
function plot1D(xx,yy,YY,err)
    [
        plot(
        [
            scatter(x=xx,y=yy, name="approx"),
            scatter(x=xx,y=YY, name="exact"),
        ]
        );
        plot(
            [
                scatter(x=xx, y= h^2*ones(N), marker_color="black", line_dash="dash", name="h^2"),
                scatter(x=xx, y= err, name="abs err")
            ],
            Layout(
                yaxis=attr(
                    type="log"
                )
            )
        )
    ]
end
function plot2D(tt, xx, u, U) 
    nt=length(tt)
    nx=length(xx)
    @assert all(size(u) .== (nt,nx))
    @assert all(size(U) .== (nt,nx))
    sigma = max(std(u),std(U))
    miny = min(minimum(u), minimum(U))-sigma
    maxy = max(maximum(u), maximum(U))+sigma
    layout = Layout(
        sliders=[attr(
            steps=[
                attr(
                    label=round(tt[n], digits=3),
                    method="restyle",
                    args=[attr(y=(u[n,:],U[n,:]))]
                )
                for n in 1:nt
            ],
            # active=1,
            currentvalue_prefix="Time: ",
            # pad_t=40
        )],
        xaxis_title="X",
        yaxis_title="U",
        yaxis_range=[miny, maxy],
        title="invisic burgers"
    )

    # we only have one trace -- a headmap for z
    t = [
        scatter(
            x=xx,
            y=u[1,:],
            name="approx"
        ),
        scatter(
            x=xx,
            y=U[1,:],
            name="exact"
        ),
    ]
    return plot(t,layout) 
    # plot(
    #         scatter(
    #             x=tt, 
    #             y=[norm(u[:,j] - U[:,j])/norm(U[:,j]) for j = 1:size(u,2)]
    #         ),
    #         Layout(
    #             xaxis_title = "space",
    #             yaxis_title="Relative Error",
    #             yaxis_type="log"
    #         )
    #     ) 
    #     plot(
    #         scatter(
    #             x=xx, 
    #             y=[norm(u[n,:] - U[n,:])/norm(U[n,:]) for n = 1:size(u,1)]
    #         ),
    #         Layout(
    #             xaxis_title = "time",
    #             yaxis_title="Relative Error",
    #             yaxis_type="log"
    #         )
    #     )
    # ]
end

function testFiniteDifMethod(
    f::Function, df::Function, dev::Union{Function,AbstractMatrix,UniformScaling{Bool}}; 
    h::AbstractVector{<:Real}=[0.1], xx::AbstractVector=[-10:0.1:10], 
    thresh::Real=10*0.1^2, debug::Bool=false
)
    DDD = length(h)
    N = length.(xx)
    IX = CartesianIndices(Tuple([1:n for n in N]))
    ff = zeros(N...)
    YY = zeros(N...)
    for ix in IX 
        in = [x[i] for (i,x) in zip(Tuple(ix),xx)]
        ff[ix] = f(in...)
        YY[ix] = df(in...)
    end
    yy = dev(ff)  
    err = zeros(N...)
    for ix in eachindex(err) 
        err[ix] = abs.(yy[ix] - YY[ix])
    end
    if debug
        if DDD == 1
            display(plot1D(xx,yy,YY,err))
        elseif DDD == 2
            display(plot2D(xx[1],xx[2],yy,YY))
        end
    end
    return maximum(err) < thresh
end
##
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
    b = y
    ##
    λ = 2
    tol=1e-4
    debugFreq= 300
    ll=Logging.Info
    maxIter=Int(1e4)
    w, fcnHist  = SINDy.lassoSolver(
        G,b, λ; tol=tol, debugFreq=debugFreq,
        ll=ll, crit=SINDy.GradNorm(), 
        maxIter=maxIter, restart=-2);
    # Correct for normalization 
    w ./= scales 
    ## check with CVX
    x     = Variable(m)
    obj   = minimize(sumsquares(G*x-b)/2 + λ*norm(x,1) )
    solve!(obj,SCS.Optimizer,silent_solver=true)
    wstar = evaluate(x)

    absErr = norm(w - wstar)
    @info "absErr= $absErr"
    return absErr < 10 * tol

end

function testGradDescent()
    n = 10
    m = n
    A = rand(n,m)
    A = A*A' + I
    b = rand(n)
    debugFreq= 10
    ll=Logging.Warn
    tol=1e-8
    f(x) = 1/2 * norm(A*x - b)^2
    grad(x) = A'*(A*x -b)
    function stepFun(x::AbstractVector,::Union{Nothing,AbstractVector}=nothing) 
        p = b - A*x
        return x + dot(p,p)/dot(p, A*p)*p
    end
    x0 = zeros(m) 
    xstar = A \ b 
    x, _ = SINDy.gradientDescent(
        f,grad,x0; stepMethod=SINDy.CustomStep(stepFun), 
        debugFreq=debugFreq,ll=ll,tol=tol);
    absErr = norm(x-xstar)  
    return absErr < tol * 10
end
##
@testset "SINDy" begin
@testset "Finite Differences 1D" begin
    h = [0.1]
    DDD = length(h)
    N = zeros(Int,DDD)
    xx = Vector(undef, DDD)
    for (i,hi) in enumerate(h)
        xx[i] = -10:hi:10
        N[i] = length(xx[i])
    end
    ## Sparse operators
    @test testFiniteDifMethod(sin, sin, 
        u->SINDy.D(N[1],h[1],0)*u; 
        h=h,xx=xx,thresh=10*h[1]^2)
    @test testFiniteDifMethod(sin, cos, 
        u->SINDy.D(N[1],h[1],1)*u; 
        h=h,xx=xx,thresh=10*h[1]^2)
    @test testFiniteDifMethod(sin, x->-sin(x), 
        u->SINDy.D(N[1],h[1],2)*u; 
        h=h,xx=xx,thresh=10*h[1]^2)
    @test testFiniteDifMethod(sin, x->-cos(x), 
        u->SINDy.D(N[1],h[1],3)*u; 
        h=h,xx=xx,thresh=10*h[1]^2)
    # Inplace / Matrix free
    @test testFiniteDifMethod(sin, sin, 
        u->SINDy.D(u,h[1],0); 
        h=h,xx=xx,thresh=10*h[1]^2)
    @test testFiniteDifMethod(sin, cos, 
        u->SINDy.D(u,h[1],1); 
        h=h,xx=xx,thresh=10*h[1]^2)
    @test testFiniteDifMethod(sin, x->-sin(x), 
        u->SINDy.D(u,h[1],2); 
        h=h,xx=xx,thresh=10*h[1]^2)
    @test testFiniteDifMethod(sin, x->-cos(x), 
        u->SINDy.D(u,h[1],3); 
        h=h,xx=xx,thresh=10*h[1]^2)
    # @test testFISTA()
end
@testset "Finite Differences 2D" begin
    U(t,x) = sin(x)*exp(-2*t)
    Ux(t,x) = cos(x)*exp(-2*t)
    Ut(t,x) = -2*sin(x)*exp(-2*t)
    Utx(t,x) = -2*cos(x)*exp(-2*t)
    ##
    ht = 0.1
    hx = 0.01
    tt = 0:ht:10
    xx = -10:hx:10
    Nt = length(tt)
    Nx = length(xx)
    ##
    @test testFiniteDifMethod(U,Ut,
        u->SINDy.D(u,ht,1,1);
        h=[ht,hx],xx=[tt,xx],thresh=10*ht^2)
    @test testFiniteDifMethod(U,Ux,
        u->SINDy.D(u,hx,1,2);
        h=[ht,hx],xx=[tt,xx],thresh=10*hx^2)
    @test testFiniteDifMethod(U,Utx,
        u->SINDy.D(SINDy.D(u,ht,1,2),hx,1,1);
        h=[ht,hx],xx=[tt,xx],thresh=10*ht^2)
end
##
@testset "Test Optimization Problem" begin 
    @test testGradDescent()
    # @test testFISTA() # failing right now...
end
end