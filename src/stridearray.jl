@inline undef_memory_buffer(::Type{T}, ::StaticInt{L}) where {T,L} =
  MemoryBuffer{L,T}(undef)
@inline undef_memory_buffer(::Type{T}, L) where {T} = Vector{T}(undef, L)
@inline undef_memory_buffer(::Type{Bit}, ::StaticInt{L}) where {L} =
  MemoryBuffer{(L + 7) >>> 3,UInt8}(undef)
@inline undef_memory_buffer(::Type{Bit}, L) = Vector{UInt8}(undef, (L + 7) >>> 3)
# @inline undef_memory_buffer(::Type{Bit}, L) = BitVector(undef, L)

struct StrideArray{S,D,T,N,C,B,R,X,O,A<:Union{MemoryBuffer,AbstractArray}} <:
       AbstractStrideArray{S,D,T,N,C,B,R,X,O}
  ptr::PtrArray{S,D,T,N,C,B,R,X,O}
  data::A
end
struct StrideBitArray{S,D,N,C,B,R,X,O,A<:Union{MemoryBuffer,AbstractArray}} <:
       AbstractStrideArray{S,D,Bool,N,C,B,R,X,O}
  ptr::BitPtrArray{S,D,N,C,B,R,X,O}
  data::A
end
StrideArray(
  ptr::BitPtrArray{S,D,N,C,B,R,X,O},
  data::A,
) where {S,D,N,C,B,R,X,O,A<:Union{MemoryBuffer,AbstractArray}} =
  StrideBitArray{S,D,N,C,B,R,X,O,A}(ptr, data)
@inline LayoutPointers.stridedpointer(A::StrideArray) = getfield(getfield(A, :ptr), :ptr)
@inline LayoutPointers.stridedpointer(A::StrideBitArray) = getfield(getfield(A, :ptr), :ptr)

const StrideVector{S,D,T,C,B,R,X,O,A} = StrideArray{S,D,T,1,C,B,R,X,O,A}
const StrideMatrix{S,D,T,C,B,R,X,O,A} = StrideArray{S,D,T,2,C,B,R,X,O,A}

const BitStrideArray{S,D,N,C,B,R,X,O} =
  Union{BitPtrArray{S,D,N,C,B,R,X,O},StrideBitArray{S,D,N,C,B,R,X,O}}

@inline StrideArray(A::AbstractArray) = StrideArray(PtrArray(A), A)

@inline function StrideArray{T}(::UndefInitializer, s::Tuple{Vararg{Union{Integer,StaticInt},N}}) where {N,T}
  x, L = calc_strides_len(T, s)
  b = undef_memory_buffer(T, L ÷ static_sizeof(T))
  # For now, just trust Julia's alignment heuristics are doing the right thing
  # to save us from having to over-allocate
  # ptr = LayoutPointers.align(pointer(b))
  StrideArray(reinterpret(Ptr{T}, pointer(b)), s, x, b, all_dense(Val{N}()))
end
@inline function StrideArray(ptr::Ptr{T}, s::S, x::X, b, ::Val{D}) where {S,X,T,D}
  StrideArray(PtrArray(ptr, s, x, Val{D}()), b)
end
@inline StrideArray(f, s::Vararg{Union{Integer,StaticInt},N}) where {N} = StrideArray{Float64}(f, s)
@inline StrideArray(f, s::Tuple{Vararg{Union{Integer,StaticInt},N}}) where {N} = StrideArray{Float64}(f, s)
@inline StrideArray(f, ::Type{T}, s::Vararg{Union{Integer,StaticInt},N}) where {T,N} = StrideArray{T}(f, s)
@inline StrideArray{T}(f, s::Vararg{Union{Integer,StaticInt},N}) where {T,N} = StrideArray{T}(f, s)
@inline function StrideArray(
  A::PtrArray{S,D,T,N},
  s::Tuple{Vararg{Union{Integer,StaticInt},N}},
) where {S,D,T,N}
  PtrArray(stridedpointer(A), s, val_dense_dims(A))
end
@inline function StrideArray(A::AbstractArray{T,N}, s::Tuple{Vararg{Union{Integer,StaticInt},N}}) where {T,N}
  StrideArray(PtrArray(stridedpointer(A), s, val_dense_dims(A)), preserve_buffer(A))
end
@inline function StrideArray{T}(f, s::Tuple{Vararg{Union{Integer,StaticInt},N}}) where {T,N}
  A = StrideArray{T}(undef, s)
  @inbounds for i ∈ eachindex(A)
    A[i] = f(T)
  end
  A
end
@inline function StrideArray{T}(::typeof(zero), s::Tuple{Vararg{Union{Integer,StaticInt},N}}) where {T,N}
  ptr = Ptr{T}(Libc.calloc(prod(s), sizeof(T)))
  A = unsafe_wrap(Array{T}, ptr, s; own = true)
  StrideArray(A)
end
mutable struct StaticStrideArray{S,D,T,N,C,B,R,X,O,L} <:
               AbstractStrideArray{S,D,T,N,C,B,R,X,O}
  data::NTuple{L,T}
  @inline StaticStrideArray{S,D,T,N,C,B,R,X,O}(
    ::UndefInitializer,
  ) where {S,D,T,N,C,B,R,X,O} =
    new{S,D,T,N,C,B,R,X,O,Int(ArrayInterface.reduce_tup(*, to_static_tuple(Val(S))))}()
  @inline StaticStrideArray{S,D,T,N,C,B,R,X,O,L}(
    ::UndefInitializer,
  ) where {S,D,T,N,C,B,R,X,O,L} = new{S,D,T,N,C,B,R,X,O,L}()
end

@generated function to_static_tuple(::Val{S}) where {S}
  t = Expr(:tuple)
  for s ∈ S.parameters
    push!(t.args, Expr(:new, s))
  end
  t
end
@inline ArrayInterface.size(::StaticStrideArray{S}) where {S} = to_static_tuple(Val(S))
@inline function ArrayInterface.strides(
  ::StaticStrideArray{S,D,T,N,C,B,R,X},
) where {S,D,T,N,C,B,R,X}
  map(Base.Fix2(>>>, intlog2(static_sizeof(T))), to_static_tuple(Val(X)))
end
@inline LayoutPointers.bytestrides(
  ::StaticStrideArray{S,D,T,N,C,B,R,X},
) where {S,D,T,N,C,B,R,X} = to_static_tuple(Val(X))
@inline ArrayInterface.offsets(
  ::StaticStrideArray{S,D,T,N,C,B,R,X,O},
) where {S,D,T,N,C,B,R,X,O} = to_static_tuple(Val(O))
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::StaticStrideArray{S,D,T}) where {S,D,T} =
  Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline Base.pointer(A::StaticStrideArray{S,D,T}) where {S,D,T} =
  Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline StrideArray{T}(::UndefInitializer, s::Tuple{Vararg{StaticInt,N}}) where {N,T} =
  StaticStrideArray{T}(undef, s)
@inline StrideArray{T}(f, s::Tuple{Vararg{StaticInt,N}}) where {N,T} =
  StaticStrideArray{T}(f, s)
@inline StrideArray{T}(::typeof(zero), s::Tuple{Vararg{StaticInt,N}}) where {N,T} =
  StaticStrideArray{T}(zero, s) # Eager when static; assumed small
@inline function StaticStrideArray{T}(
  ::UndefInitializer,
  s::Tuple{Vararg{StaticInt,N}},
) where {N,T}
  StaticStrideArray{T}(undef, s, all_dense(Val(N)))
end
@inline function LayoutPointers.bytestrideindex(
  A::StaticStrideArray{S,D,T,N,C,B,R},
) where {S,D,T,N,C,B,R}
  StrideIndex{N,R,C}(LayoutPointers.bytestrides(A), offsets(A))
end

@inline function StaticStrideArray{T}(
  ::UndefInitializer,
  s::Tuple{Vararg{StaticInt,N}},
  ::Val{D},
) where {N,T,D}
  x, L = calc_strides_len(T, s)
  R = ntuple(Int, Val(N))
  O = ntuple(_ -> One(), Val(N))
  Tshifter = intlog2(static_sizeof(T))
  StaticStrideArray{typeof(s),D,T,N,1,0,R,typeof(x),typeof(O),Int(L >>> Tshifter)}(undef)
end

function dense_quote(N::Int, b::Bool)
  d = Expr(:tuple)
  for n = 1:N
    push!(d.args, b)
  end
  Expr(:call, Expr(:curly, :Val, d))
end
@generated all_dense(::Val{N}) where {N} = dense_quote(N, true)

@generated function calc_strides_len(::Type{T}, s::Tuple{Vararg{StaticInt,N}}) where {T,N}
  L = Base.allocatedinline(T) ? sizeof(T) : sizeof(Int)
  t = Expr(:tuple)
  for n ∈ 1:N
    push!(t.args, StaticInt{L}())
    l::Int = (s.parameters[n].parameters[1])::Int
    if (T ≢ Bit) || (n ≠ 1)
      L *= l
    else
      L *= ((l + 7) & -8)
    end
  end
  Expr(:tuple, t, StaticInt{L}())
end
function calc_strides_len(::Type{T}, ::Tuple{}) where {T}
end
@generated function calc_strides_len(::Type{T}, s::Tuple{Vararg{Union{Integer,StaticInt},N}}) where {T,N}
  last_sx = :s_0
  st = Base.allocatedinline(T) ? sizeof(T) : sizeof(Int)
  q = Expr(:block, Expr(:meta, :inline), Expr(:(=), last_sx, StaticInt{st}()))
  t = Expr(:tuple)
  for n ∈ 1:N
    push!(t.args, last_sx)
    new_sx = Symbol(:s_, n)
    sz_expr = Expr(:call, getfield, :s, n, false)
    ((T === Bit) && (n == 1)) &&
      (sz_expr = :(($sz_expr + StaticInt{7}()) & StaticInt{-8}()))
    push!(q.args, Expr(:(=), new_sx, Expr(:call, *, last_sx, sz_expr)))
    last_sx = new_sx
  end
  push!(q.args, Expr(:tuple, t, last_sx))
  q
end
@inline function StaticStrideArray{T}(f, s::Tuple{Vararg{StaticInt,N}}) where {T,N}
  A = StaticStrideArray{T}(undef, s)
  @inbounds for i ∈ eachindex(A)
    A[i] = f(T)
  end
  A
end

@inline LayoutPointers.preserve_buffer(A::MemoryBuffer) = A
@inline LayoutPointers.preserve_buffer(A::StrideArray) = preserve_buffer(getfield(A, :data))

@inline PtrArray(A::Union{StrideArray,StrideBitArray}) = getfield(A, :ptr)

@inline maybe_ptr_array(A) = A
@inline maybe_ptr_array(A::AbstractArray) = maybe_ptr_array(ArrayInterface.device(A), A)
@inline maybe_ptr_array(::ArrayInterface.CPUPointer, A::AbstractArray) = PtrArray(A)
@inline maybe_ptr_array(_, A::AbstractArray) = A

@inline ArrayInterface.size(A::Union{StrideArray,StrideBitArray}) =
  getfield(getfield(A, :ptr), :size)

@inline LayoutPointers.bytestrides(A::Union{StrideArray,StrideBitArray}) =
  bytestrides(getfield(getfield(A, :ptr), :ptr))
@inline ArrayInterface.strides(A::Union{StrideArray,StrideBitArray}) =
  strides(getfield(A, :ptr))
@inline ArrayInterface.offsets(A::Union{StrideArray,StrideBitArray}) =
  offsets(getfield(getfield(A, :ptr), :ptr))

@inline zeroindex(r::ArrayInterface.OptionallyStaticUnitRange{One}) =
  CloseOpen(Zero(), last(r))
@inline zeroindex(r::Base.OneTo) = CloseOpen(Zero(), last(r))
@inline zeroindex(r::AbstractUnitRange) = Zero():(last(r)-first(r))

@inline zeroindex(r::CloseOpen{Zero}) = r
@inline zeroindex(r::ArrayInterface.OptionallyStaticUnitRange{Zero}) = r
@inline zeroindex(A::PtrArray{S,D}) where {S,D} =
  PtrArray(zstridedpointer(A), size(A), Val{D}())
@inline zeroindex(A::Union{StrideArray,StrideBitArray}) =
  StrideArray(zeroindex(PtrArray(A)), preserve_buffer(A))
@inline zeroindex(A::StaticStrideArray) = StrideArray(zeroindex(PtrArray(A)), A)

@generated rank_to_sortperm_val(::Val{R}) where {R} = :(Val{$(rank_to_sortperm(R))}())
@inline function similar_layout(A::AbstractStrideArray{S,D,T,N,C,B,R}) where {S,D,T,N,C,B,R}
  permutedims(similar(permutedims(A, rank_to_sortperm_val(Val{R}()))), Val{R}())
end
@inline function similar_layout(A::AbstractArray)
  b = preserve_buffer(A)
  GC.@preserve b begin
    similar_layout(PtrArray(A))
  end
end
@inline function Base.similar(A::AbstractStrideArray{S,D,T}) where {S,D,T}
  StrideArray{T}(undef, size(A))
end
@inline function Base.similar(A::BitPtrArray)
  StrideArray{Bit}(undef, size(A))
end
@inline function Base.similar(A::AbstractStrideArray, ::Type{T}) where {T}
  StrideArray{T}(undef, size(A))
end


@inline function Base.view(
  A::AbstractStrideArray,
  i::Vararg{Union{Integer,AbstractRange,Colon},K},
) where {K}
  StrideArray(view(PtrArray(A), i...), preserve_buffer(A))
end

@inline function zview(
  A::AbstractStrideArray,
  i::Vararg{Union{Integer,AbstractRange,Colon},K},
) where {K}
  StrideArray(zview(PtrArray(A), i...), preserve_buffer(A))
end
@inline function Base.permutedims(A::AbstractStrideArray, ::Val{P}) where {P}
  StrideArray(permutedims(PtrArray(A), Val{P}()), preserve_buffer(A))
end
@inline Base.adjoint(a::AbstractStrideVector) =
  StrideArray(adjoint(PtrArray(a)), preserve_buffer(a))


function gc_preserve_call(ex, skip = 0)
  q = Expr(:block)
  call = Expr(:call, esc(ex.args[1]))
  gcp = Expr(:gc_preserve, call)
  for i ∈ 2:length(ex.args)
    arg = ex.args[i]
    if i + 1 ≤ skip
      push!(call.args, arg)
      continue
    end
    A = gensym(:A)
    buffer = gensym(:buffer)
    if arg isa Expr && arg.head === :kw
      push!(call.args, Expr(:kw, arg.args[1], Expr(:call, :maybe_ptr_array, A)))
      arg = arg.args[2]
    else
      push!(call.args, Expr(:call, :maybe_ptr_array, A))
    end
    push!(q.args, :($A = $(esc(arg))))
    push!(q.args, Expr(:(=), buffer, Expr(:call, :preserve_buffer, A)))
    push!(gcp.args, buffer)
  end
  push!(q.args, gcp)
  q
end
"""
    @gc_preserve foo(A, B, C)

  Apply to a single, non-nested, function call. It will `GC.@preserve` all the arguments, and substitute suitable arrays with `PtrArray`s.
  This has the benefit of potentially allowing statically sized mutable arrays to be both stack allocated, and passed through a non-inlined function boundary.
  """
macro gc_preserve(ex)
  @assert ex.head === :call
  gc_preserve_call(ex)
end


@inline LayoutPointers.zero_offsets(A::AbstractArray) =
  StrideArray(LayoutPointers.zero_offsets(PtrArray(A)), A)

@generated function Base.IndexStyle(
  ::Type{<:Union{BitPtrArray{S,D,N,C,B,R},StrideBitArray{S,D,N,C,B,R}}},
) where {S,D,N,C,B,R}
  # if is column major || is a transposed contiguous vector
  if all(D) && (
    (isone(C) && R === ntuple(identity, Val(N))) ||
    (C === 2 && R === (2, 1) && S <: Tuple{One,Integer})
  )
    if N > 1
      ks1 = known(S)[1]
      if ks1 === nothing || ((ks1 & 7) != 0)
        return :(IndexCartesian())
      end
    end
    :(IndexLinear())
  else
    :(IndexCartesian())
  end
end
