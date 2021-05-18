"""
    rank_to_sortperm(::NTuple{N,Int}) -> NTuple{N,Int}
Returns the `sortperm` of the stride ranks.
"""
function rank_to_sortperm(R::NTuple{N,Int}) where {N}
    sp = ntuple(zero, Val{N}())
    r = ntuple(n -> sum(R[n] .≥ R), Val{N}())
    @inbounds for n = 1:N
        sp = Base.setindex(sp, n, r[n])
    end
    sp
end
rank_to_sortperm(R) = sortperm(R)


# @inline function gep_no_offset(ptr::VectorizationBase.AbstractStridedPointer, i::Tuple)
    # VectorizationBase.gep(pointer(ptr), VectorizationBase.tdot(ptr, i, VectorizationBase.strides(ptr), VectorizationBase.nopromote_axis_indicator(ptr)))
# end
# @inline function similar_with_offset(sptr::StridedPointer{T,N,C,B,R,X,O}, ptr::Ptr{T}) where {T,N,C,B,R,X,O}
#     StridedPointer{T,N,C,B,R,X}(ptr, sptr.strd, zerotuple(Val{N}()))
# end
function view_quote(i, K, S, D, T, N, C, B, R, X, O, zero_offsets::Bool = false)
    @assert ((K == N) || isone(K))

    inds = Expr(:tuple)
    Nnew = 0
    s = Expr(:tuple)
    x = Expr(:tuple)
    o = Expr(:tuple)
    Rnew = Expr(:tuple)
    Dnew = Expr(:tuple)
    Cnew = -1
    Bnew = ifelse(iszero(B), 0, -1)
    sortp = rank_to_sortperm(R)
    still_dense = true
    densev = Vector{Bool}(undef, K)
    for k ∈ 1:K
        iₖ = Expr(:ref, :i, k)
        if i[k] === Colon
            Nnew += 1
            push!(inds.args, Expr(:ref, :o, k))
            push!(s.args, Expr(:ref, :s, k))
            push!(x.args, Expr(:ref, :x, k))
            push!(o.args, zero_offsets ? :(Zero()) : :(One()))
            if k == C
                Cnew = Nnew
            end
            if k == B
                Bnew = Nnew
            end
            push!(Rnew.args, R[k])
        else
            push!(inds.args, Expr(:call, :first, iₖ))
            if i[k] <: AbstractRange
                Nnew += 1
                push!(s.args, Expr(:call, :static_length, iₖ))
                push!(x.args, Expr(:ref, :x, k))
                push!(o.args, zero_offsets ? :(Zero()) : :(One()))
                if k == C
                    Cnew = Nnew
                end
                if k == B
                    Bnew = Nnew
                end
                push!(Rnew.args, R[k])
            end
        end
        spₙ = sortp[k]
        still_dense &= D[spₙ]
        if still_dense# && (D[spₙ])
            ispₙ = i[spₙ]
            still_dense = (ispₙ <: AbstractUnitRange) || (ispₙ === Colon)
            densev[spₙ] = still_dense
            if still_dense
                still_dense = if ((ispₙ === Colon)::Bool || (ispₙ <: Base.Slice)::Bool)
                    true
                else
                    ispₙ_len = ArrayInterface.known_length(ispₙ)
                    if ispₙ_len !== nothing
                        _sz = getfield(S, :parameters)[spₙ]
                        if _sz <: StaticInt
                            ispₙ_len == getfield(_sz, :parameters)[1]
                        else
                            false
                        end
                    else
                        false
                    end
                    false
                end
            end
        else
            densev[spₙ] = false
        end
    end
    for k ∈ 1:K
        iₖt = i[k]
        if (iₖt === Colon) || (iₖt <: AbstractVector)
            push!(Dnew.args, densev[k])
        end
    end    
    quote
        $(Expr(:meta,:inline))
        sp = stridedpointer(A)
        s = size(A)
        x = strides(sp)
        o = offsets(sp)
        new_sp = StridedPointer{$T,$Nnew,$Cnew,$Bnew,$Rnew}(gep(sp, $inds), $x, $o)
        PtrArray(new_sp, $s, Val{$Dnew}())
    end
end

@generated function Base.view(A::PtrArray{S,D,T,N,C,B,R,X,O}, i::Vararg{Union{Integer,AbstractRange,Colon},K}) where {K,S,D,T,N,C,B,R,X,O}
    view_quote(i, K, S, D, T, N, C, B, R, X, O)
end
@generated function zview(A::PtrArray{S,D,T,N,C,B,R,X,O}, i::Vararg{Union{Integer,AbstractRange,Colon},K}) where {K,S,D,T,N,C,B,R,X,O}
    view_quote(i, K, S, D, T, N, C, B, R, X, O, true)
end

@inline function Base.vec(A::PtrArray{S,D,T,N,C,0}) where {S,D,T,N,C}
    @assert all(D) "All dimensions must be dense for a vec view. Try `vec(copy(A))` instead."
    sp = StridedPointer(pointer(A), (VectorizationBase.static_sizeof(T),), (One(),))
    PtrArray(sp, (static_length(A),), Val((true,)))
end

