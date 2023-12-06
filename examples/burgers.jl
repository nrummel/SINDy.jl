using ProgressMeter, Distributions, PlotlyJS
using SparseArrays, LinearAlgebra, BenchmarkTools
using SINDy
function genData(; c::Real=1.1, 
    μ::Real=0.1,
    Xmin::Real=-2*pi,
    Xmax::Real=2*pi,
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
    dg = (x,t,u)-> 1 + μ*t* df(x -(c + μ * u)*t); 
    # Double check that f is behaving
    nt = length(tt);
    nx = length(xx);
    u = zeros(nt,nx);
    u[1,:] = f.(xx);

    ## LaxWendroff
    # J(u) = c + μ*u
    # for n = 2:nt 
    #     for j = 1:nx 
    #         ujm1 = j == 1 ? 0 : u[n-1,j-1]
    #         ujp1 = j == nx ? 0 : u[n-1,j+1]
    #         Ap = J(1/2* (u[n-1,j]+ujp1))
    #         Am = J(1/2* (u[n-1,j]+ujm1))
    #         u[n,j] = (
    #             u[n-1,j] 
    #             - ht / (2*hx) * (ujp1 -  ujm1) 
    #             + ht^2 /(2*hx) * (
    #                 Ap * (ujp1 -  u[n-1,j])  
    #                 - Am * (u[n-1,j] -  ujm1))
    #         )
    #     end
    # end
    #
    tol= 1e-12;
    maxIter = 1000;
    warnFlag = 0
    pbar = Progress(nt-1)
    for n = 2:nt
        next!(pbar; showvalues = [(:iter,n)])
        Threads.@threads for j = 1:nx 
            # perform newton on root finding problem for nonlinear eq
            ukm1 = u[n-1,j];
            uk = zeros(nx);
            for k = 1:maxIter
                uk = ukm1 - g(xx[j], tt[n], ukm1) / dg(xx[j], tt[n], ukm1);
                err = abs(uk - ukm1);
                if err < tol 
                    break
                elseif k == maxIter
                    warnFlag += 1
                end
                ukm1 = uk
            end
            u[n,j]= uk;
        end
    end
    warnFlag != 0 && (@warn "Max iterations met $warnFlag times")
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
        yaxis_title="Absolute Error",
        yaxis_range=[miny, maxy],
        title="Invisic Burgers"
    )

    # we only have one trace -- a headmap for z
    t = [
        scatter(
            x=xx,
            y=u[1,:],
            name="Approximation"
        ),
        scatter(
            x=xx,
            y=U[1,:],
            name="Exact Solution"
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
#Get Data
alpha = [3,3]
P = 3
Q = 0
c = 2 
μ = 0.1
hx=0.2
ht=0.01
tt, xx, u, nx, nt, hx, ht = genData(c=c,μ=μ,hx=hx,ht=ht,Xmax=20);
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
    findfirst(names .== "Dt:0:Dx1:1:p2")
]

wstar[ixExact] = [-c, -μ/2]
F(w) = norm(G*w-y)
@info "==================================================="
@info "=============== Inviscide Burgers  ================"
@info "==================================================="
@info "$(round.(w',sigdigits=2))"
@info "obj lasso = $(F(w))"
@info "$(round.(w_mp',sigdigits=2))"
@info "obj matching pursuit = $(F(w_mp))"
@info "$(round.(wstar',sigdigits=2))"
@info "obj truth = $(F(wstar))"
##
Dxu = SINDy.D(u,hx,1,2)
ix = findfirst(names .== "Dt:0:Dx1:1:p1")
DxU = reshape(G[:,ix],nt,nx)

Dxu2 = SINDy.D(u.^2,hx,1,2)
ix = findfirst(names .== "Dt:0:Dx1:1:p2")
DxU2 = reshape(G[:,ix],nt,nx)
u + c *Dxu + μ/2*Dxu2 

Dtu =  SINDy.D(nt, ht, 1) * u
DtU = reshape(y,nt,nx)

plot2D(tt,xx,  
abs.(Dtu - reshape(G * w_mp, nt,nx)) ,
abs.(DtU +( c *DxU + μ/2*DxU2) ),
)
## 
NN = Int(1e3)
X = rand(NN,NN)
##
# D_mat = SINDy.D(nx,hx,1)'
Dx_sparse = u-> u* SINDy.D(nx,hx,1)'
Dx = u-> SINDy.D(u,hx,1,2)
@info "Sparse Matrix implementation"
@btime Dx_sparse(u)
@info "Matrix Free implementation"
Dx(u)
nothing