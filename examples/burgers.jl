using ProgressMeter, Distributions, PlotlyJS
using SINDy
function genData(; c::Real=1.1, 
    μ::Real=0.1,
    Xmin::Real=-10.0,
    Xmax::Real=10.0,
    T::Real=5,
    hx::Real=0.1,
    ht::Real=0.01
)
    # defin the initial condition
    # UnitGaussian = Normal(0,1)
    # f = x-> pdf(UnitGaussian, x);
    # df = x-> (x-1) / sqrt(2 *pi) * exp(-(x-1)^2/2);
    f = x-> sin(x);
    df = x-> cos(x);

    tt = 0:ht:T;
    xx = Xmin:hx:Xmax;
    nx = length(xx);
    nt = length(tt);
    ## Generate data for invisic burgers
    g = (x,t,u)-> u - f(x -(c + μ  * u)*t);
    dg = (x,t,u)-> 1 + μ * df(x -(c + μ * u)*t); 
    # Double check that f is behaving
    nt = length(tt);
    nx = length(xx);

    #
    tol= 1e-12;
    maxIter = 1000;
    u = zeros(nt,nx);
    u[1,:] = f.(xx);

    p = Progress(nt-1)
    warnFlag = true
    for n = 2:nt
        next!(p; showvalues = [(:iter,n)])
        for j = 1:nx 
            # perform newton on root finding problem for nonlinear eq
            ukm1 = u[n-1,j];
            uk = zeros(nx);
            for k = 1:maxIter
                uk  = ukm1 - g(xx[j], tt[n], ukm1) / dg(xx[j], tt[n], ukm1);
                err = abs(uk - ukm1) / abs(uk);
                if err < tol 
                    break
                elseif warnFlag && k == maxIter
                    @warn "Max iterations met"
                    warnFlag =false 
                end
            end
            u[n,j]= uk;
        end
    end
    return tt, xx, u, nx, nt, hx, ht
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

function plotRelErr(tt,xx,u, U) 
    [
        plot(
            scatter(
                x=tt, 
                y=[norm(u[:,n] - U[:,n])/norm(U[:,n]) for n = 1:size(u,2)]
            ),
            Layout(
                xaxis_title = "time",
                yaxis_title="Relative Error",
                yaxis_type="log"
            )
        );
        plot(
            scatter(
                x=xx, 
                y=[norm(u[j,:] - U[j,:])/norm(U[j,:]) for j = 1:size(u,1)]
            ),
            Layout(
                xaxis_title = "space",
                yaxis_title="Relative Error",
                yaxis_type="log"
            )
        );
    ]
end
## Get Data
alpha = [3,3]
P = 3
Q = 0
c = 2 
μ = 0.1
hx=0.1
ht=0.01
tt, xx, u, nx, nt, hx, ht = genData(c=c,μ=μ,Xmax=20.0,hx=hx,ht=ht);
plot2D(tt,xx,u,zeros(size(u)))
##
@info "Building Lib ..."
Lib = SINDy.buildLib(alpha,[nt,nx],[ht,hx],P,Q)
@info "Building Linear System ..."
G,y,names = SINDy.buildLinearSystem(Lib, u)
@info "Solving with LASSO ..."
w = SINDy.solveWithLasso(G, y, 0)
@info "Solving with MP ..."
w_mp = SINDy.MatchingPursuit(G, y, 2)
wstar = zeros(size(G,2))
ixExact = [
    findfirst(names .== "Dt:0:Dx1:1:p1"), 
    findfirst(names .== "Dt:0:Dx1:2:p2")
]
wstar[ixExact] = [-c, -μ/2]
F(w) = norm(G*w-y)
@info "==================================================="
@info "=============== Inviscide Burgers  ================"
@info "==================================================="
@info "$(round.(w',sigdigits=2))"
@info "obj lasso = $(F(w))"
@info "$(round.(w_mp',sigdigits=2))"
@info "obj matching pursuit = $(F(w))"
@info "$(round.(wstar',sigdigits=2))"
@info "obj truth = $(F(wstar))"
## 
Dx_sparse = u-> u*SINDy.D(nx,hx,1)'
Dx = u-> SINDy.D(u,hx,1,2)
Dx2 = u-> mydev(u,hx,1,2)
@time Dx_sparse(u)
@time Dx(u)
@time Dx2(u)
nothing