
# `Reference`s will automatically be unpacked.
mutable struct Reference{T}
    data::T
end

@inline function ThreadingUtilities.load(p::Ptr{UInt}, ::Type{Reference{T}}, i) where {T}
    r = Base.unsafe_pointer_to_objref(reinterpret(Ptr{Reference{T}}, p + sizeof(UInt)*(i += 1)))
    i, getfield(r, :data)
end
@inline function ThreadingUtilities.store!(p::Ptr{UInt}, r::Reference, i)
    ThreadingUtilities.store!(p + sizeof(UInt)*(i += 1), reinterpret(UInt, pointer_from_objref(r)))
    i
end

@generated _isbits(::T) where {T} = Expr(:call, isbitstype(T) ? :True : :False)

@inline object_and_preserve(x) = _object_and_preserve(x, _isbits(x))
@inline _object_and_preserve(x, ::True) = (x, nothing)
@inline function _object_and_preserve(x, ::False)
    r = Reference(x)
    r, r
end

@inline object_and_preserve(A::AbstractArray{T}) where {T<:VectorizationBase.NativeTypes} = array_object_and_preserve(ArrayInterface.device(A), A)
@inline object_and_preserve(A::AbstractArray) = _object_and_preserve(A, False())
@inline array_object_and_preserve(::ArrayInterface.CPUPointer, A::AbstractArray) = (PtrArray(A), preserve_buffer(A))
@inline array_object_and_preserve(_, A::AbstractArray) = _object_and_preserve(A, False())




