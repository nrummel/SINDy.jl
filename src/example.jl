using ProgressMeter
using Distributions
using PlotlyJS
function genData()
    ## Define data
    Xmin = -10
    Xmax = 10
    hx = 0.1
    T = 10
    ht =0.1
    c = 1; 
    μ = 0.1;
    # defin the initial condition
    UnitGaussian = Normal(0,1)
    f = x-> pdf(UnitGaussian, x);
    df = x-> (x-1) / sqrt(2 *pi) * exp(-(x-1)^2/2);

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
function plotTestData(tt,xx, uB)

    layout = Layout(
        sliders=[attr(
            steps=[
                attr(
                    label=round(tt[i], digits=3),
                    method="restyle",
                    args=[attr(y=(uB[:, i],))]
                )
                for i in 1:nt
            ],
            # active=1,
            currentvalue_prefix="Time: ",
            # pad_t=40
        )],
        xaxis_title="X",
        yaxis_title="U",
        title="invisic burgers"
    )

    # we only have one trace -- a headmap for z
    t = scatter(
        x=xx,
        y=uB[:,1],
    )
    return plot(t, layout)
end