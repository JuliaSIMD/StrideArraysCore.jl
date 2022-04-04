
@generated function _object_and_preserve(x::T) where {T}
  # q = if sizeof(T) > 64
  # Expr(:block, :(rx = $(ManualMemory.Reference)(x)), :((rx, rx)))
  # elseif isbitstype(T)
  q = isbitstype(T) ? :((x, nothing)) : :((x, x))
  Expr(:block, Expr(:meta, :inline), q)
end
@inline object_and_preserve(x) = _object_and_preserve(x)

@inline object_and_preserve(x::String) = (x, x)
@inline object_and_preserve(A::AbstractArray{T}) where {T<:NativeTypes} =
  array_object_and_preserve(ArrayInterface.device(A), A)
@inline object_and_preserve(A::AbstractArray) = (A, A)
@inline array_object_and_preserve(::ArrayInterface.CPUPointer, A::AbstractArray) =
  (PtrArray(A), preserve_buffer(A))
@inline array_object_and_preserve(_, A::AbstractArray) = _object_and_preserve(A)

@inline object_and_preserve(r::Base.RefValue{T}) where {T} =
  (Base.unsafe_convert(Ptr{T}, r), r)

@inline function object_and_preserve(t::Tuple{T}) where {T}
  o, p = object_and_preserve(only(t))
  (o,), (p,)
end
@inline function object_and_preserve(t::Tuple{T1,T2,Vararg}) where {T1,T2}
  oh, ph = object_and_preserve(first(t))
  ot, pt = object_and_preserve(Base.tail(t))
  (oh, ot...), (ph, pt...)
end
