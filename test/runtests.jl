using StrideArraysCore, ThreadingUtilities, Aqua
using Test

@testset "StrideArraysCore.jl" begin

    Aqua.test_all(StrideArraysCore)

    @testset "StrideArrays Basic" begin
        A = rand(100, 100);
        B = copy(A);
        C = StrideArraysCore.PtrArray(A);
        GC.@preserve A begin
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
        end
        W = rand(2,3,4);
        X = PtrArray(W);
        GC.@preserve W begin
            @test W == X
            @test permutedims(W, (1,2,3)) == permutedims(X, Val((1,2,3)))
            @test permutedims(W, (1,3,2)) == permutedims(X, Val((1,3,2)))
            @test permutedims(W, (2,1,3)) == permutedims(X, Val((2,1,3)))
            @test permutedims(W, (2,3,1)) == permutedims(X, Val((2,3,1)))
            @test permutedims(W, (3,1,2)) == permutedims(X, Val((3,1,2)))
            @test permutedims(W, (3,2,1)) == permutedims(X, Val((3,2,1)))
            @test_throws BoundsError X[length(X) + 1]
            @test_throws BoundsError X[-4]
            @test_throws BoundsError X[2,5,3]
        end
        y = rand(77);
        GC.@preserve y begin
            z = PtrArray(y);
            @test y == z
            @test pointer(y) === pointer(z)
            @test_throws BoundsError z[-8]
            @test_throws BoundsError z[88]
        end
    end
    @testset "ThreadingUtilities" begin
        xu = zeros(UInt, 100);
        x = rand(100); y = rand(100); z = rand(100);
        t = (x,y,z)
        r = StrideArraysCore.Reference(t)
        GC.@preserve xu x y z begin
            ThreadingUtilities.store!(pointer(xu), r, 0)
            @test ThreadingUtilities.load(pointer(xu), typeof(r), 0) === (1, t)
            for a ∈ t
                p, g = StrideArraysCore.object_and_preserve(a)
                @test g === a
                @test p isa PtrArray
                @test pointer(p) === pointer(a)
                @test p !== a
                @test p == a
            end
        end
    end
end
