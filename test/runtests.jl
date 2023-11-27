using SINDy
using Test, PlotlyJS
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
    m = 10 
    A = rand(n,m)
    y = rand(n)
    λ = 0
    wstar = (A'*A) \ A'*b 
    errThresh=1e-8
    w  = fista(A,y, λ;errThresh=errThresh)
    # w = (w ./ weights[IX] ) * weights[1] 
    return norm(w - wstar)/norm(wstar) < 10 * errThresh
end
@testset "SINDy.jl" begin
    @test testFiniteDiffMethod(sin, cos, SINDy.D)
    @test testFiniteDiffMethod(sin, x->-sin(x), SINDy.D2)
    @test testFiniteDiffMethod(sin, x->-cos(x), SINDy.D3)
    @test testFISTA()
end
