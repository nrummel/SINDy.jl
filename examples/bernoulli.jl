using ProgressMeter, Distributions, PlotlyJS
using SINDy, Symbolics

""" Bernoulli's equation with constant coefficients
U' = -p U + q U^2, U(0) = U0
⇒ u(t) = 1 / (q/p + (1 - q/p * U0)/ U0 * exp(p*t)) 
"""
U(t, p::Real, q::Real, U0::Real) = 1/ (q/p + (1 - q/p *U0)/U0 *exp(p*t)) 
function buildExactLib(alpha, P, p, q, u0)
    @variables t
    y = U(t,p,q,u0)
    Lib = Vector{NamedTuple}(undef,(alpha+1)*P)
    ix = 0
    for i = 1:alpha+1
        for j = 1:P
            ix += 1
            tmp = nothing
            if i == 1 
                tmp = eval(build_function(y^j,t))
            else 
                d=Differential(t)^(i-1)
                tmp = eval(build_function(expand_derivatives(d(y^j)),t))
            end
            Lib[ix] =(
                name= "Dt:$(i-1):p$j",
                fun=tt->tmp.(tt)
            )
        end
    end
   
    return Lib
end

function genBernoulli(p::Real, q::Real, u0::Real; 
    T::Real=10.0,
    h::Real=0.1
)
    tt = 0:h:T;
    nt = length(tt); 
    y0 = 0;
    
    u = U.(tt,p,q,u0)
    return tt, nt, h, u
end
##
function plot1D(tt,u, U, title="") 
    plot(
        [
            scatter(
                x=tt,
                y=u,
                name="Approximation"
            ),
            scatter(
                x=tt,
                y=U,
                name="Exact"
            ),
        ],
        Layout(title=title,
        yaxis_title="Absolute Error",
        xaxis_title="time")
    )
end
function plotRelErr(tt,u, U) 
    plot(
        scatter(
            x=tt, 
            y=abs.(u - U) ./ abs.(U)
        ),
        Layout(
            xaxis_title = "time",
            yaxis_title="Relative Error",
            yaxis_type="log"
        )
    )
end
#
alpha = 3
P = 3
Q = 0
p = -0.01
q = 3
u0 = -1
T = 3
ht = 0.001
tt, nt, ht, u = genBernoulli(p, q, u0;T=T,h=ht);
approxLib = SINDy.buildLib(alpha,nt,ht,P,Q)
exactLib = buildExactLib(alpha,P,p,q,u0)
G_approx,y_approx,names_approx = SINDy.buildLinearSystem(approxLib, u)
G_exact,y_exact,names_exact = SINDy.buildLinearSystem(exactLib,tt)
w_approx = SINDy.solveWithLasso(G_approx, y_approx, 2)
w_exact = SINDy.solveWithLasso(G_exact, y_exact, 2)
w_approx_mp = SINDy.MatchingPursuit(G_approx, y_approx, 2)
w_exact_mp = SINDy.MatchingPursuit(G_exact, y_exact, 2)
wstar = zeros(size(G_exact,2))
wstar[1:2] = [-p, q]
F_approx(w) = norm(G_approx*w-y_approx)
F_exact(w) = norm(G_exact*w-y_exact)
##
@info "==================================================="
@info "============= With Finite Difference =============="
@info "==================================================="
@info "$(round.(w_approx',sigdigits=2))"
@info "obj lasso = $(F_approx(w_approx))"
@info "$(round.(w_approx_mp',sigdigits=2))"
@info "obj matching pursuit = $(F_approx(w_approx))"
@info "$(round.(wstar',sigdigits=2))"
@info "obj truth = $(F_approx(wstar))"
@info "==================================================="
@info "============= With Exact Derivatives =============="
@info "==================================================="
@info "$(round.(w_exact',sigdigits=2))"
@info "obj best = $(F_exact(w_exact))"
@info "$(round.(w_exact_mp',sigdigits=2))"
@info "obj best = $(F_exact(w_exact_mp))"
@info "$(round.(wstar',sigdigits=2))"
@info "obj truth = $(F_exact(wstar))"
##

plot1D(
    tt, 
    abs.(y_approx - G_approx*wstar),
    abs.(y_approx- G_approx*w_approx_mp),
    # title="Bernoulli"
)