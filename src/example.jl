using ProgressMeter, Distributions, PlotlyJS
using SINDy, Convex, ECOS, SCS
function genData(; c::Real=1.1, 
    μ::Real=0.1,
    Xmin::Real=-10.0,
    Xmax::Real=10.0,
    T::Real=5,
    hx::Real=0.1,
    ht::Real=0.1
)
    # defin the initial condition
    UnitGaussian = Normal(0,1)
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
    dg = (x,t,u)-> 1 + μ* df(x -(c + μ * u)*t); 
    # Double check that f is behaving
    nt = length(tt);
    nx = length(xx);

    #
    tol= 1e-6;
    maxIter = 1000;
    uB = zeros(nx,nt);
    uB[:,1] = f.(xx);

    p = Progress(nt-1)
    for n = 2:nt
        next!(p; showvalues = [(:iter,n)])
        for j = 1:nx 
            # perform newton on root finding problem for nonlinear eq
            ukm1 = uB[j,n-1];
            uk = 0;
            for k = 1:maxIter
                uk  = ukm1 - g(xx[j], tt[n], ukm1) / dg(xx[j], tt[n], ukm1);
                err = abs(uk - ukm1) / abs(uk);
                if err < tol 
                    break
                end  
            end
            uB[j,n]= uk;
        end
    end
    return tt, xx, uB, nx, nt, hx, ht
end
function plotTestData(tt,xx, u, U) 
    nt=length(tt)
    nx=length(xx)
    sigma = max(std(u),std(U))
    miny = min(minimum(u), minimum(U))-sigma
    maxy = max(maximum(u), maximum(U))+sigma
    layout = Layout(
        sliders=[attr(
            steps=[
                attr(
                    label=round(tt[i], digits=3),
                    method="restyle",
                    args=[attr(y=(u[:, i],U[:,i]))]
                )
                for i in 1:nt
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
            y=u[:,1],
            name="approx"
        ),
        scatter(
            x=xx,
            y=U[:,1],
            name="exact"
        ),
    ]
    return plot(t, layout)
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
c = 2 
μ = 0.1
tt, xx, uB, nx, nt, hx, ht = genData(c=c,μ=μ,Xmax=20.0);
plotTestData(tt,xx,uB, zeros(nx,nt))
## 
(SpatialDevLib,TemporalDevLib) = SINDy.getDevLibs(3,3,nx,nt,hx,ht)
FunctionLib = SINDy.getFunctionLib(3,0);
## Build Optimization problem
II = length(FunctionLib)
J = length(SpatialDevLib)
K = length(TemporalDevLib)
M = prod((nx-4)*(nt-4))
N = II*J*K
G = zeros(M, N)
p = Progress(N)
Ix2Name = Vector{String}(undef, N)
for (i, (tDevName, tdev)) in enumerate(TemporalDevLib)
    for (j, (sDevName, sdev)) in enumerate(SpatialDevLib)
        for (k, (funName, fun)) in enumerate(FunctionLib)
            next!(p; showvalues = [(:sdev,sDevName),(:tdev,tDevName), (:fun, funName)])
            ix = II*J*(i-1) + II*(j-1) + k
            Ix2Name[ix] = sDevName*":"*tDevName*":"*funName
            G[:,ix] = (sdev*fun.(uB)*tdev)[:]
        end
    end
end
ix = findfirst(Ix2Name .== "space:0:time:1:p1")
y = G[:,ix]
G = G[:,setdiff(1:N, ix)]
Ix2Name = Ix2Name[setdiff(1:N, ix)]
##
ixwrong = findfirst(Ix2Name .== "space:0:time:1:p2")
ixright= findfirst(Ix2Name .== "space:1:time:0:p2")
plotTestData(tt[3:end-2],xx[3:end-2],reshape(G[:,ixwrong],nx-4,nt-4),-reshape(G[:,ixright],nx-4,nt-4))
## Precondition G 
# fact = svd(G) 
# C = diagm(1 ./ fact.S)* fact.U' # preconditioner
# Cinv = fact.U * diagm(fact.S)
# A = C*G
# b = C*y
scales = zeros(N-1)
A = copy(G) 
for i in 1:N-1
    scales[i] = std(A[:,i])
    A[:,i] = (A[:,i] .- mean(A[:,i])) / scales[i]
end
b = y
## Lasso 
λ = 2
x = Variable(N-1)
x.value = A'*b
obj = minimize(norm(A*x-b) + λ*norm(x,1) )
solve!(obj,SCS.Optimizer)
alpha1 = evaluate(x)
w1 = alpha1 
nzIx = findall(abs.(w1) .> 1e-3)
# Could use CG or if small enough direct method... 
x = Variable(length(nzIx))
obj = minimize(norm(A[:,nzIx]*x-b) )
solve!(obj,SCS.Optimizer)
alpha2 = evaluate(x)
w2 = zeros(N-1)
for (i,ix) in enumerate(nzIx)
    w2[ix] = alpha2[i] 
end
w2 ./= scales
##
F(x) = 1/2*norm(G*x - y)^2
wstar = zeros(N-1)
ixExact = [
    findfirst(Ix2Name .== "space:1:time:0:p1"), 
    findfirst(Ix2Name .== "space:1:time:0:p2")
]
wstar[ixExact] = [-c, -μ/2]
@info "$(Ix2Name[nzIx])"
@info "$(w2[nzIx])"
@info "obj best = $(F(w2))"
@info "$(Ix2Name[ixExact])"
@info "$(wstar[ixExact])"
@info "obj truth = $(F(wstar))"
nothing
##
# Test Finie Differences 
U(x,t) = sin(x)*exp(-2*t)
Ux(x,t) = cos(x)*exp(-2*t)
Ut(x,t) = -2*sin(x)*exp(-2*t)
Utx(x,t) = -2*cos(x)*exp(-2*t)
U2x(x,t) = 2*sin(x)*cos(x)*exp(-4*t)
U2t(x,t) = -4*sin(x)^2*exp(-4*t)
Dx = SINDy.D(nx,hx,1)
Dt = SINDy.D(nt,ht,1)
u = zeros(nx,nt)
ux = zeros(nx-4,nt-4)
ut = zeros(nx-4,nt-4)
utx= zeros(nx-4,nt-4)
u2x= zeros(nx-4,nt-4)
u2t= zeros(nx-4,nt-4)
for (j,x) in enumerate(xx), (n,t) in enumerate(tt)
    u[j,n] = U(x,t)
end
for (j,x) in enumerate(xx[3:end-2]), (n,t) in enumerate(tt[3:end-2])
    ux[j,n] = Ux(x,t)
    ut[j,n] = Ut(x,t)
    utx[j,n] = Utx(x,t)
    u2x[j,n] = U2x(x,t)
    u2t[j,n] = U2t(x,t)
end
##
uxhat = Dx*u;
uthat = u*Dt'
utxhat = Dx*u*Dt'
u2xhat = Dx*(u.^2)
u2that = (u.^2)*Dt'

plotTestData(
    tt[3:end-2],xx[3:end-2],
    utxhat, utx)
plotRelErr(tt[3:end-2],xx[3:end-2],
    utxhat,utx)