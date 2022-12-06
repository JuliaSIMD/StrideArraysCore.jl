@inline function Base.view(
  A::AbstractPtrArray{T,N,R,S,X,O,P},
  i::Vararg{Union{Integer,AbstractRange,Colon},N},
) where {T,N,R,S,X,O,P}
  PtrArray(SubArray(A, Base.to_indices(A, i)))
end
@inline function zview(
  A::AbstractPtrArray{T,N,R,S,X,O,P},
  i::Vararg{Union{Integer,AbstractRange,Colon},N},
) where {T,N,R,S,X,O,P}
  zero_offsets(view(A, i...))
end

@inline Base.view(A::AbstractPtrArray, ::Colon) = vec(A)
@inline zview(A::AbstractPtrArray, ::Colon) = vec(A)

@inline Base.view(A::AbstractPtrArray{<:Any,N}, ::Vararg{Colon,N}) where {N} = A
@inline zview(A::AbstractPtrArray{<:Any,N}, ::Vararg{Colon,N}) where {N} = A

@inline Base.view(A::AbstractPtrVector, ::Colon) = A
@inline zview(A::AbstractPtrVector, ::Colon) = A

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
