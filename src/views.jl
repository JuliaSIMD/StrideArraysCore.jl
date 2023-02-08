@inline function Base.view(
  A::AbstractPtrArray{T,N},
  i::Vararg{Union{Integer,AbstractRange,Colon},N}
) where {T,N}
  PtrArray(SubArray(A, Base.to_indices(A, i)))
end
@inline function Base.view(
  A::AbstractPtrArray{T,N},
  i::AbstractUnitRange
) where {T,N}
  view(vec(A), i)
end
@inline function Base.view(
  A::AbstractPtrArray{T,1},
  i::AbstractUnitRange
) where {T}
  sx = stride(A, static(1))
  p = pointer(A) + first(i) * sizeof(T) * sx
  PtrArray(p, (length(i),), (sx,))
end

@inline function Base.view(
  B::BitPtrArray{N},
  i::Vararg{Union{Integer,AbstractRange,Colon},N}
) where {N}
  A = SubArray(B, Base.to_indices(B, i))
  p = _offset_ptr(stridedpointer(B), i)
  sz = size(A)
  sx = _sparse_strides(dense_dims(A), strides(A))
  R = map(Int, stride_rank(A))
  PtrArray(p, sz, sx, offsets(A), _compact_rank(Val(R)))
end
@inline function Base.view(
  B::BitPtrArray{N},
  i::Vararg{Union{Integer,AbstractRange,Colon},M}
) where {N,M}
  A = SubArray(B, i...)
  # TODO: handle offsetting correctly
  # `SubArray` is off by a factor 8 too large
  # rather than calculate the correct pointer offset..a fac
  p8off = LayoutPointers.memory_reference(A)[1]
  pb = pointer(B)
  off8 = UInt(p8off - pb)
  off = off8 >>> 3
  @assert off8 == (off << 3)
  p = pb + off
  sz = size(A)
  sx = _sparse_strides(dense_dims(A), strides(A))
  R = map(Int, stride_rank(A))
  PtrArray(p, sz, sx, offsets(A), _compact_rank(Val(R)))
end
@inline function zview(
  A::AbstractPtrArray{T,N,R,S,X,O,P},
  i::Vararg{Union{Integer,AbstractRange,Colon},M}
) where {T,N,R,S,X,O,P,M}
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
