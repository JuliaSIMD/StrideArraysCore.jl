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
end
