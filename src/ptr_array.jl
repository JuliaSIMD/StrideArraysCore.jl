
abstract type AbstractStrideArray{S,D,T,N,C,B,R,X,O} <: DenseArray{T,N} end
abstract type AbstractPtrStrideArray{S,D,T,N,C,B,R,X,O} <: AbstractStrideArray{S,D,T,N,C,B,R,X,O} end
const AbstractStrideVector{S,D,T,C,B,R,X,O} = AbstractStrideArray{S,D,T,1,C,B,R,X,O}
const AbstractStrideMatrix{S,D,T,C,B,R,X,O} = AbstractStrideArray{S,D,T,2,C,B,R,X,O}

struct PtrArray{S,D,T,N,C,B,R,X,O} <: AbstractPtrStrideArray{S,D,T,N,C,B,R,X,O}
    ptr::StridedPointer{T,N,C,B,R,X,O}
    size::S
end
@inline function PtrArray(ptr::StridedPointer{T,N,C,B,R,X,O}, size::S, ::Val{D}) where {S,D,T,N,C,B,R,X,O}
    PtrArray{S,D,T,N,C,B,R,X,O}(ptr, size)
end

const PtrVector{S,D,T,C,B,R,X,O} = PtrArray{S,D,T,1,C,B,R,X,O}
const PtrMatrix{S,D,T,C,B,R,X,O} = PtrArray{S,D,T,2,C,B,R,X,O}

@inline PtrArray(A::AbstractArray) = PtrArray(stridedpointer(A), size(A), val_dense_dims(A))

@inline VectorizationBase.stridedpointer(A::PtrArray) = getfield(A, :ptr)
@inline Base.pointer(A::AbstractStrideArray) = pointer(stridedpointer(A))
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::AbstractStrideArray) where {T} = Base.unsafe_convert(Ptr{T}, pointer(A))
@inline Base.elsize(::AbstractStrideArray{<:Any,<:Any,T}) where {T} = sizeof(T)

@inline ArrayInterface.size(A::PtrArray) = getfield(A, :size)
@inline VectorizationBase.bytestrides(A::PtrArray) = getfield(getfield(A, :ptr), :strd)
ArrayInterface.device(::AbstractStrideArray) = ArrayInterface.CPUPointer()

ArrayInterface.contiguous_axis(::Type{<:AbstractStrideArray{S,D,T,N,C}}) where {S,D,T,N,C} = StaticInt{C}()
ArrayInterface.contiguous_batch_size(::Type{<:AbstractStrideArray{S,D,T,N,C,B}}) where {S,D,T,N,C,B} = ArrayInterface.StaticInt{B}()

static_expr(N::Int) = Expr(:call, Expr(:curly, :StaticInt, N))
static_expr(b::Bool) = Expr(:call, b ? :True : :False)
@generated function ArrayInterface.stride_rank(::Type{<:AbstractStrideArray{S,D,T,N,C,B,R}}) where {S,D,T,N,C,B,R}
    t = Expr(:tuple)
    for r ∈ R
        push!(t.args, static_expr(r::Int))
    end
    t
end
@generated function ArrayInterface.dense_dims(::Type{<:AbstractStrideArray{S,D}}) where {S,D}
    t = Expr(:tuple)
    for d ∈ D
        push!(t.args, static_expr(d::Bool))
    end
    t
end

@inline bytestride(A, n) = VectorizationBase.bytestrides(A)[n]
function default_stridedpointer_quote(::Type{T}, N, Ot) where {T}
    C = 1
    B = 0
    R = Expr(:tuple)
    o = Expr(:tuple)
    for n ∈ 1:N
        push!(R.args, n)
        push!(o.args, Expr(:call, Ot))
    end
    quote
        $(Expr(:meta,:inline))
        StridedPointer{$T,$N,$C,$B,$R}(ptr, x, $o)
    end
end

@generated function default_stridedpointer(ptr::Ptr{T}, x::X) where {T, N, X <: Tuple{Vararg{Integer,N}}}
    default_stridedpointer_quote(T, N, :One)
end
@generated function default_zerobased_stridedpointer(ptr::Ptr{T}, x::X) where {T, N, X <: Tuple{Vararg{Integer,N}}}
    default_stridedpointer_quote(T, N, :Zero)
end
@inline function ptrarray0(ptr::Ptr{T}, s::Tuple{Vararg{Integer,N}}, x::Tuple{Vararg{Integer,N}}, ::Val{D}) where {T,N,D}
    PtrArray(default_zerobased_stridedpointer(ptr, x), s, Val{D}())
end
@inline function PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{Integer,N}}, x::Tuple{Vararg{Integer,N}}, ::Val{D}) where {T,N,D}
    PtrArray(default_stridedpointer(ptr, x), s, Val{D}())
end

function ptrarray_densestride_quote(::Type{T}, N, stridedpointer_offsets) where {T}
    last_sx = :s_0
    q = Expr(:block, Expr(:meta,:inline), Expr(:(=), last_sx, static_sizeof(T)))
    t = Expr(:tuple); d = Expr(:tuple);
    n = 0
    while true
        n += 1
        push!(t.args, last_sx)
        push!(d.args, true)
        n == N && break
        new_sx = Symbol(:s_,n)
        push!(q.args, Expr(:(=), new_sx, Expr(:call, *, last_sx, Expr(:call, GlobalRef(Core,:getfield), :s, n, false))))
        last_sx = new_sx
    end
    push!(q.args, :(PtrArray($stridedpointer_offsets(ptr, $t), s, Val{$d}())))
    q
end
@generated function PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{Integer,N}}) where {T,N}
    ptrarray_densestride_quote(T, N, :default_stridedpointer)
end
@generated function ptrarray0(ptr::Ptr{T}, s::Tuple{Vararg{Integer,N}}) where {T,N}
    ptrarray_densestride_quote(T, N, :default_zerobased_stridedpointer)
end

@generated function ArrayInterface.strides(A::PtrArray{S,D,T,N}) where {S,D,T,N}
  size_T = Base.allocatedinline(T) ? sizeof(T) : sizeof(Int)
  shifter = static_expr(VectorizationBase.intlog2(size_T))
  x = Expr(:tuple)
  for n in 1:N
    push!(x.args, Expr(:call, :(>>>), Expr(:ref, :x, n), shifter))
  end
  quote
    $(Expr(:meta,:inline))
    x = A.ptr.strd
    $x
  end
end

@inline Base.size(A::AbstractStrideArray) = map(Int, size(A))
@inline Base.strides(A::AbstractStrideArray) = map(Int, strides(A))

@inline create_axis(s, ::Zero) = CloseOpen(s)
@inline function create_axis(s, ::One)
  VectorizationBase.assume(s ≥ 0)
  Base.OneTo(s)
end
@inline create_axis(s, o) = CloseOpen(o, s+o)

@inline ArrayInterface.axes(A::AbstractStrideArray) = map(create_axis, size(A), offsets(A))
@inline Base.axes(A::AbstractStrideArray) = axes(A)

@inline ArrayInterface.offsets(A::PtrArray) = getfield(getfield(A, :ptr), :offsets)
@inline ArrayInterface.static_length(A::AbstractStrideArray) = prod(size(A))

# type stable, because index known at compile time
@inline type_stable_select(t::NTuple, ::StaticInt{N}) where {N} = t[N]
@inline type_stable_select(t::Tuple, ::StaticInt{N}) where {N} = t[N]
# type stable, because tuple is homogenous
@inline type_stable_select(t::NTuple, i::Integer) = t[i]
# make the tuple homogenous before indexing
@inline type_stable_select(t::Tuple, i::Integer) = map(Int, t)[i]

@inline function ArrayInterface._axes(A::AbstractStrideArray{S,D,T,N}, i::Integer) where {S,D,T,N}
  if i ≤ N
    o = type_stable_select(offsets(A), i)
    s = type_stable_select(size(A), i)
    return create_axis(s, o)
  else
    return One():One()
  end
end
@inline Base.axes(A::AbstractStrideArray, i::Integer) = axes(A, i)

@inline function ArrayInterface.size(A::AbstractStrideVector, i::Integer)
    d = Int(length(A))
    ifelse(isone(i), d, one(d))
end
@inline ArrayInterface.size(A::AbstractStrideVector, ::StaticInt{N}) where {N} = One()
@inline ArrayInterface.size(A::AbstractStrideVector, ::StaticInt{1}) = length(A)
@inline ArrayInterface.size(A::AbstractStrideArray, ::StaticInt{N}) where {N} = size(A)[N]
@inline ArrayInterface.size(A::AbstractStrideArray, i::Integer) = type_stable_select(size(A), i)
@inline Base.size(A::AbstractStrideArray, i::Integer) = size(A, i)


# Base.IndexStyle(::Type{<:AbstractStrideArray}) = IndexCartesian()
# Base.IndexStyle(::Type{<:AbstractStrideVector{<:Any,<:Any,<:Any,1}}) = IndexLinear()
@generated function Base.IndexStyle(::Type{<:AbstractStrideArray{S,D,T,N,C,B,R}}) where {S,D,T,N,C,B,R}
    # if is column major || is a transposed contiguous vector
    if all(D) && ((isone(C) && R === ntuple(identity, Val(N))) || (C === 2 && R === (2,1) && S <: Tuple{One,Integer}))
        :(IndexLinear())
    else
        :(IndexCartesian())
    end          
end

@inline VectorizationBase.preserve_buffer(A::PtrArray) = nothing


@generated function pload(p::Ptr{T}) where {T}
  if Base.allocatedinline(T)
    Expr(:block, Expr(:meta,:inline), :(unsafe_load(p)))
  else
    Expr(:block, Expr(:meta,:inline), :(ccall(:jl_value_ptr, Ref{$T}, (Ptr{Cvoid},), unsafe_load(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, p)))))
  end
end
@generated function pstore!(p::Ptr{T}, v::T) where {T}
  if Base.allocatedinline(T)
    Expr(:block, Expr(:meta,:inline), :(unsafe_store!(p, v); return nothing))
  else
    Expr(:block, Expr(:meta,:inline), :(unsafe_store!(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, p), Base.pointer_from_objref(v)); return nothing))
  end
end
@inline pstore!(p::Ptr{T}, v) where {T} = pstore!(p, convert(T, v))

function rank2sortperm(R)
  map(R) do r
    sum(map(≥(r),R))
  end
end
# @generated function _offset_ptr(ptr::AbstractStridedPointer{T,N,C,B,R}, i::Tuple{Vararg{Integer,NI}}) where {T,N,C,B,R,NI}
#   if N ≠ NI
#     if (N > NI) & (NI ≠ 1)
#       throw(ArgumentError("If the dimension of the array exceeds the dimension of the index, then the index should be linear/one dimensional."))
#     end
#     # use only the first index. Supports, for example `x[i,1,1,1,1]` when `x` is a vector, or `A[i]` where `A` is an array with dim > 1.
#     return Expr(:block, Expr(:meta,:inline), :(pointer(ptr) + (first(i)-1)*$(static_sizeof(T))))
#   end
#   sp = rank2sortperm(R)
#   q = Expr(:block, Expr(:meta,:inline), :(p = pointer(ptr)), :(o = VectorizationBase.offsets(ptr)), :(x = strides(ptr)))
#   gf = GlobalRef(Core,:getfield)
#   for n ∈ 1:N
#     j = findfirst(==(n),sp)::Int
#     index = Expr(:call, gf, :i, j, false)
#     offst = Expr(:call, gf, :o, j, false)
#     strid = Expr(:call, gf, :x, j, false)
#     push!(q.args, :(p += ($index - $offst)*$strid))
#   end
#   q
# end
@generated function _offset_ptr(ptr::AbstractStridedPointer{T,N,C,B,R}, i::Tuple{Vararg{Integer,NI}}) where {T,N,C,B,R,NI}
  N == 0 && return Expr(:block, Expr(:meta,:inline), :(pointer(ptr)))
  if N ≠ NI
    if (N > NI) & (NI ≠ 1)
      throw(ArgumentError("If the dimension of the array exceeds the dimension of the index, then the index should be linear/one dimensional."))
    end
    # use only the first index. Supports, for example `x[i,1,1,1,1]` when `x` is a vector, or `A[i]` where `A` is an array with dim > 1.
    return Expr(:block, Expr(:meta,:inline), :(pointer(ptr) + (first(i)-1)*$(static_sizeof(T))))
  end
  sp = rank2sortperm(R)
  q = Expr(:block, Expr(:meta,:inline), :(p = pointer(ptr)), :(o = VectorizationBase.offsets(ptr)), :(x = strides(ptr)))
  gf = GlobalRef(Core,:getfield)
  for n ∈ 1:N
    j = findfirst(==(n),sp)::Int
    index = Expr(:call, gf, :i, j, false)
    offst = Expr(:call, gf, :o, j, false)
    strid = Expr(:call, gf, :x, j, false)
    push!(q.args, :(p += ($index - $offst)*$strid))
  end
  q
end
# @inline _offset_ptr(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,Zero}}, i::Tuple{Vararg{Integer,NI}}) where {T,N,C,B,R,NI,X} = __offset_ptr(ptr,i)
# @inline function _offset_ptr(ptr::AbstractStridedPointer{T,N,C,B,R}, i::Tuple{Vararg{Integer,NI}}) where {T,N,C,B,R,NI}
#   __offset_ptr(VectorizationBase.center(ptr), i)
# end

# Base.@propagate_inbounds Base.getindex(A::AbstractStrideVector, i::Int, j::Int) = A[i]
@inline function Base.getindex(A::PtrArray, i::Vararg{Integer})
  @boundscheck checkbounds(A, i...)
  pload(_offset_ptr(stridedpointer(A), i))
end
@inline function Base.getindex(A::AbstractStrideArray, i::Vararg{Integer,K}) where {K}
  b = preserve_buffer(A)
  P = PtrArray(A)
  GC.@preserve b begin
    @boundscheck checkbounds(P, i...)
    pload(_offset_ptr(stridedpointer(A), i))
  end
end
@inline function Base.setindex!(A::PtrArray, v, i::Vararg{Integer,K}) where {K}
  @boundscheck checkbounds(A, i...)
  pstore!(_offset_ptr(stridedpointer(A), i), v)
  v
end
@inline function Base.setindex!(A::AbstractStrideArray, v, i::Vararg{Integer,K}) where {K}
  b = preserve_buffer(A)
  P = PtrArray(A)
  GC.@preserve b begin
    @boundscheck checkbounds(P, i...)
    pstore!(_offset_ptr(stridedpointer(A), i), v)
  end
  v
end
@inline function Base.getindex(A::PtrArray{S,D,T}, i::Integer) where {S,D,T}
  @boundscheck checkbounds(A, i)
  pload(pointer(A) + (i-oneunit(i))*static_sizeof(T))
end
@inline function Base.getindex(A::AbstractStrideArray{S,D,T}, i::Integer) where {S,D,T}
  b = preserve_buffer(A)
  P = PtrArray(A)
  GC.@preserve b begin
    @boundscheck checkbounds(P, i)
    pload(pointer(A) + (i-oneunit(i))*static_sizeof(T))
  end
end
@inline function Base.setindex!(A::PtrArray{S,D,T}, v, i::Integer) where {S,D,T}
  @boundscheck checkbounds(A, i)
  pstore!(pointer(A) + (i-oneunit(i))*static_sizeof(T), v)
  v
end
@inline function Base.setindex!(A::AbstractStrideArray{S,D,T}, v, i::Integer) where {S,D,T}
  b = preserve_buffer(A)
  P = PtrArray(A)
  GC.@preserve b begin
    @boundscheck checkbounds(P, i)
    pstore!(pointer(A) + (i-oneunit(i))*static_sizeof(T), v)
  end
  v
end


