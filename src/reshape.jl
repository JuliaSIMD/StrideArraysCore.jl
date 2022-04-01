@inline Base.vec(A::PtrArray{S,D,T,1,C,0}) where {S,D,T,C} = A
@inline function Base.vec(A::PtrArray{S,D,T,N,C,0}) where {S,D,T,N,C}
  @assert all(D) "All dimensions must be dense for a vec view. Try `vec(copy(A))` instead."
  si = StrideIndex{1,(1,),1}((static_sizeof(T),), (One(),))
  sp = stridedpointer(pointer(A), si)
  PtrArray(sp, (static_length(A),), Val((true,)))
end
@inline Base.vec(A::StaticStrideArray{S,D,T,1}) where {S,D,T} = A
@inline Base.vec(A::StaticStrideArray{S,D,T,N}) where {S,D,T,N} = StrideArray(vec(PtrArray(A)), A)
@inline Base.vec(A::StrideArray) = StrideArray(vec(PtrArray(A)),preserve_buffer(A))

@inline function Base.reshape(A::PtrArray{S,D}, dims::Tuple{Vararg{Integer}}) where {S,D}
  @assert all(D) "All dimensions must be dense for a reshaped view. Try `reshape(copy(A),...)` instead."
  PtrArray(pointer(A), dims)
end
@inline function Base.reshape(A::PtrArray{S,D}, dims::Tuple{Integer,Vararg{Integer}}) where {S,D}
  @assert all(D) "All dimensions must be dense for a reshaped view. Try `reshape(copy(A),...)` instead."
  PtrArray(pointer(A), dims)
end
@inline function Base.reshape(A::PtrArray{S,D}, dims::Tuple{Vararg{Int}}) where {S,D}
  @assert all(D) "All dimensions must be dense for a reshaped view. Try `reshape(copy(A),...)` instead."
  PtrArray(pointer(A), dims)
end
@inline function Base.reshape(A::StrideArray, dims::Tuple{Vararg{Integer}})
  StrideArray(reshape(PtrArray(A), dims), preserve_buffer(A))
end
@inline function Base.reshape(A::StrideArray, dims::Tuple{Integer,Vararg{Integer}})
  StrideArray(reshape(PtrArray(A), dims), preserve_buffer(A))
end
@inline function Base.reshape(A::StrideArray, dims::Tuple{Vararg{Int}})
  StrideArray(reshape(PtrArray(A), dims), preserve_buffer(A))
end


