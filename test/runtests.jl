using StrideArraysCore
using Test

@testset "StrideArraysCore.jl" begin

    A = rand(100, 100);
    B = copy(A);
    C = StrideArraysCore.PtrArray(A);
    @test A == B
    C .*= 3
    @test A == 3 .* B
    @test C == A
    @test C == 3 .* B

    D = copy(A);
    Cslice = view(C, 23:48, 17:89)
    Cslice .= 2
    @test D != C
    D[23:48,17:89] .= 2
    @test D == C

    @test C  isa PtrArray
    @test C' isa PtrArray
    @test permutedims(C, Val((2,1))) isa PtrArray
    @test C' == D'

    W = rand(2,3,4);
    X = PtrArray(W);
    @test W == X
    @test permutedims(W, (1,2,3)) == permutedims(X, Val((1,2,3)))
    @test permutedims(W, (1,3,2)) == permutedims(X, Val((1,3,2)))
    @test permutedims(W, (2,1,3)) == permutedims(X, Val((2,1,3)))
    @test permutedims(W, (2,3,1)) == permutedims(X, Val((2,3,1)))
    @test permutedims(W, (3,1,2)) == permutedims(X, Val((3,1,2)))
    @test permutedims(W, (3,2,1)) == permutedims(X, Val((3,2,1)))

    y = rand(77); z = PtrArray(y);
    @test y == z
    @test pointer(y) === pointer(z)
end
