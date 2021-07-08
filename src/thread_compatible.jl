

@generated _isbits(::T) where {T} = Expr(:call, isbitstype(T) ? :True : :False)
@inline object_and_preserve(x) = _object_and_preserve(x, _isbits(x))
@inline _object_and_preserve(x, ::True) = (x, nothing)
@inline _object_and_preserve(x, ::False) = (x, x)

@inline object_and_preserve(A::AbstractArray{T}) where {T<:VectorizationBase.NativeTypes} = array_object_and_preserve(ArrayInterface.device(A), A)
@inline object_and_preserve(A::AbstractArray) = _object_and_preserve(A, False())
@inline array_object_and_preserve(::ArrayInterface.CPUPointer, A::AbstractArray) = (PtrArray(A), preserve_buffer(A))
@inline array_object_and_preserve(_, A::AbstractArray) = _object_and_preserve(A, False())

@inline object_and_preserve(r::Base.RefValue{T}) where {T} = (Base.unsafe_convert(Ptr{T}, r), r)

@inline function object_and_preserve(t::Tuple{T}) where {T}
  o, p = object_and_preserve(only(t))
  (o,), (p,)
end
@inline function object_and_preserve(t::Tuple{T1,T2,Vararg}) where {T1,T2}
  oh, ph = object_and_preserve(first(t))
  ot, pt = object_and_preserve(Base.tail(t))
  (oh, ot...), (ph, pt...)
end

