abstract type AbstractStrideArray{S,D,T,N,C,B,R,X,O} <: DenseArray{T,N} end
abstract type AbstractPtrStrideArray{S,D,T,N,C,B,R,X,O} <:
              AbstractStrideArray{S,D,T,N,C,B,R,X,O} end
const AbstractStrideVector{S,D,T,C,B,R,X,O} = AbstractStrideArray{S,D,T,1,C,B,R,X,O}
const AbstractStrideMatrix{S,D,T,C,B,R,X,O} = AbstractStrideArray{S,D,T,2,C,B,R,X,O}

struct PtrArray{S,D,T,N,C,B,R,X,O} <: AbstractPtrStrideArray{S,D,T,N,C,B,R,X,O}
  ptr::StridedPointer{T,N,C,B,R,X,O}
  size::S
end
@inline function PtrArray(
  ptr::StridedPointer{T,N,C,B,R,X,O},
  size::S,
  ::Val{D},
) where {S,D,T,N,C,B,R,X,O}
  PtrArray{S,D,T,N,C,B,R,X,O}(ptr, size)
end
struct BitPtrArray{S,D,N,C,B,R,X,O} <: AbstractPtrStrideArray{S,D,Bool,N,C,B,R,X,O}
  ptr::StridedBitPointer{N,C,B,R,X,O}
  size::S
end
@inline function PtrArray(
  ptr::StridedBitPointer{N,C,B,R,X,O},
  size::S,
  ::Val{D},
) where {S,D,N,C,B,R,X,O}
  BitPtrArray{S,D,N,C,B,R,X,O}(ptr, size)
end

const PtrVector{S,D,T,C,B,R,X,O} = PtrArray{S,D,T,1,C,B,R,X,O}
const PtrMatrix{S,D,T,C,B,R,X,O} = PtrArray{S,D,T,2,C,B,R,X,O}

@inline PtrArray(A::AbstractArray) = PtrArray(stridedpointer(A), size(A), val_dense_dims(A))

@inline LayoutPointers.stridedpointer(A::AbstractPtrStrideArray) = getfield(A, :ptr)
@inline Base.pointer(A::AbstractStrideArray) = pointer(stridedpointer(A))
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::AbstractStrideArray) where {T} =
  Base.unsafe_convert(Ptr{T}, pointer(A))
@inline Base.elsize(::AbstractStrideArray{<:Any,<:Any,T}) where {T} = sizeof(T)

@inline ArrayInterface.size(A::AbstractPtrStrideArray) = getfield(A, :size)
@inline LayoutPointers.bytestrides(A::AbstractPtrStrideArray) =
  bytestrides(getfield(A, :ptr))
ArrayInterface.device(::AbstractStrideArray) = ArrayInterface.CPUPointer()

ArrayInterface.contiguous_axis(::Type{<:AbstractStrideArray{S,D,T,N,C}}) where {S,D,T,N,C} =
  StaticInt{C}()
ArrayInterface.contiguous_batch_size(
  ::Type{<:AbstractStrideArray{S,D,T,N,C,B}},
) where {S,D,T,N,C,B} = ArrayInterface.StaticInt{B}()

ArrayInterface.known_size(::Type{<:AbstractStrideArray{S}}) where {S} = Static.known(S)

@generated function ArrayInterface.stride_rank(
  ::Type{<:AbstractStrideArray{S,D,T,N,C,B,R}},
) where {S,D,T,N,C,B,R}
  t = Expr(:tuple)
  for r ∈ R
    push!(t.args, StaticInt{r}())
  end
  t
end
@generated function ArrayInterface.dense_dims(
  ::Type{<:AbstractStrideArray{S,D}},
) where {S,D}
  t = Expr(:tuple)
  for d ∈ D
    if d
      push!(t.args, True())
    else
      push!(t.args, False())
    end
  end
  t
end

# @inline bytestride(A, n) = LayoutPointers.bytestrides(A)[n]

function onetupleexpr(N::Int)
  t = Expr(:tuple)
  for _ = 1:N
    push!(t.args, One())
  end
  Expr(:block, Expr(:meta, :inline), t)
end
@generated onetuple(::Val{N}) where {N} = onetupleexpr(N)

@inline function default_strideindex(
  s::Tuple{Vararg{Integer,N}},
  o::Tuple{Vararg{Integer,N}},
) where {N}
  StrideIndex{N,ntuple(identity, Val(N)),1}(s, o)
end
@inline function default_stridedpointer(
  ptr::Ptr{T},
  x::X,
) where {T,N,X<:Tuple{Vararg{Integer,N}}}
  stridedpointer(ptr, default_strideindex(x, onetuple(Val(N))))
end
@inline function default_zerobased_stridedpointer(
  ptr::Ptr{T},
  x::X,
) where {T,N,X<:Tuple{Vararg{Integer,N}}}
  stridedpointer(ptr, default_strideindex(x, LayoutPointers.zerotuple(Val(N))))
end

@inline function ptrarray0(
  ptr::Ptr{T},
  s::Tuple{Vararg{Integer,N}},
  x::Tuple{Vararg{Integer,N}},
  ::Val{D},
) where {T,N,D}
  PtrArray(default_zerobased_stridedpointer(ptr, x), s, Val{D}())
end
@inline function PtrArray(
  ptr::Ptr{T},
  s::Tuple{Vararg{Integer,N}},
  x::Tuple{Vararg{Integer,N}},
  ::Val{D},
) where {T,N,D}
  PtrArray(default_stridedpointer(ptr, x), s, Val{D}())
end

@inline LayoutPointers.zero_offsets(A::AbstractPtrStrideArray{S,D}) where {S,D} =
  PtrArray(LayoutPointers.zero_offsets(stridedpointer(A)), size(A), Val{D}())



function ptrarray_densestride_quote(::Type{T}, knowns, N, stridedpointer_offsets) where {T}
  last_sx = :s_0
  isdense = true
  last_ksx = sizeof(T)
  q = Expr(:block, Expr(:meta, :inline))#, Expr(:(=), last_sx, static_sizeof(T)))
  t = Expr(:tuple)
  d = Expr(:tuple)
  n = 0
  while true
    n += 1
    if last_ksx != 0
      push!(t.args, static(last_ksx))
    else
      push!(t.args, last_sx)
    end
    push!(d.args, isdense)
    n == N && break
    curs = knowns[n]
    if (curs === nothing) | (last_ksx == 0)
      szn = Expr(:call, getfield, :s, n, false)
      new_sx = Symbol(:s_, n)
      last_sx_expr = if last_ksx != 0
        # first unknown dimension
        _last_ksx = last_ksx
        last_ksx = 0
        if T === Bit && n == 1
          isdense = false
          Expr(:call, &, Expr(:call, +, szn, static(7)), static(-8))
        else
          Expr(:call, *, _last_ksx, szn)
        end
      elseif T === Bit && n == 1
        Expr(:call, &, Expr(:call, +, szn, static(7)), static(-8))
      else
        Expr(:call, *, last_sx, szn)
      end
      push!(q.args, Expr(:(=), new_sx, last_sx_expr))
      last_sx = new_sx
    elseif T === Bit && n == 1
      padded = (curs + 7) & -8
      isdense = curs == padded
      last_ksx *= padded
    else
      last_ksx *= curs
    end
  end
  push!(q.args, :(PtrArray($stridedpointer_offsets(ptr, $t), s, Val{$d}())))
  q
end
@generated function PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{Integer,N}}) where {T,N}
  ptrarray_densestride_quote(T, known(s), N, :default_stridedpointer)
end
@generated function ptrarray0(ptr::Ptr{T}, s::Tuple{Vararg{Integer,N}}) where {T,N}
  ptrarray_densestride_quote(T, known(s), N, :default_zerobased_stridedpointer)
end

intlog2(N::I) where {I<:Integer} = (8sizeof(I) - one(I) - leading_zeros(N)) % I
intlog2(::Type{T}) where {T} = intlog2(static_sizeof(T))
@generated intlog2(::StaticInt{N}) where {N} =
  Expr(:call, Expr(:curly, :StaticInt, intlog2(N)))

@inline function ArrayInterface.strides(A::PtrArray{S,D,T,N}) where {S,D,T,N}
  map(Base.Fix2(>>>, intlog2(static_sizeof(T))), bytestrides(A))
end
@inline function ArrayInterface.strides(A::BitPtrArray{S,D,N}) where {S,D,N}
  bytestrides(A)
end

@inline Base.size(A::AbstractStrideArray) = map(Int, size(A))
@inline Base.strides(A::AbstractStrideArray) = map(Int, strides(A))

@inline create_axis(s, ::Zero) = CloseOpen(s)
@inline create_axis(s, ::One) = Base.OneTo(unsigned(s))
@inline create_axis(::StaticInt{N}, ::One) where {N} = One():StaticInt{N}()
@inline create_axis(s, o) = CloseOpen(o, s + o)

@inline ArrayInterface.axes(A::AbstractStrideArray) = map(create_axis, size(A), offsets(A))
@inline Base.axes(A::AbstractStrideArray) = axes(A)

@generated function ArrayInterface.axes_types(
  ::Type{<:AbstractStrideArray{S,D,T,N,C,B,R,X,O}},
) where {S,D,T,N,C,B,R,X,O}
  s = known(S)
  o = known(O)
  t = Expr(:curly, :Tuple)
  for i in eachindex(s, o)
    oi = o[i]
    si = s[i]
    if oi === nothing
      push!(t.args, CloseOpen{Int,Int})
    elseif oi == 0
      si = s[i]
      if si === nothing
        push!(t.args, CloseOpen{StaticInt{0},Int})
      else
        @assert si isa Int
        push!(t.args, CloseOpen{StaticInt{0},StaticInt{si}})
      end
    elseif oi == 1
      if si === nothing
        push!(t.args, Base.OneTo{UInt})
      else
        @assert si isa Int
        push!(t.args, ArrayInterface.OptionallyStaticUnitRange{StaticInt{1},StaticInt{si}})
      end
    else
      @assert oi isa Int
      if si === nothing
        push!(t.args, CloseOpen{StaticInt{oi},Int})
      else
        @assert si isa Int
        push!(t.args, CloseOpen{StaticInt{oi},StaticInt{oi + si}})
      end
    end
  end
  t
end


@inline ArrayInterface.offsets(A::PtrArray) = offsets(getfield(A, :ptr))
@inline ArrayInterface.offsets(A::BitPtrArray) = offsets(getfield(A, :ptr))
@inline ArrayInterface.static_length(A::AbstractStrideArray) =
  ArrayInterface.reduce_tup(*, size(A))

# type stable, because index known at compile time
@inline type_stable_select(t::NTuple, ::StaticInt{N}) where {N} = getfield(t, N, false)
@inline type_stable_select(t::Tuple, ::StaticInt{N}) where {N} = getfield(t, N, false)
# type stable, because tuple is homogenous
@inline type_stable_select(t::NTuple, i::Integer) = getfield(t, i, false)
# make the tuple homogenous before indexing
@inline type_stable_select(t::Tuple, i::Integer) = getfield(map(Int, t), i, false)

@inline ArrayInterface._axes(A::AbstractStrideArray{S,D,T,N}, i::Integer) where {S,D,T,N} =
  __axes(A, i)
@inline ArrayInterface._axes(A::AbstractStrideArray{S,D,T,N}, i::Int) where {S,D,T,N} =
  __axes(A, i)
@inline ArrayInterface._axes(
  A::AbstractStrideArray{S,D,T,N},
  ::StaticInt{I},
) where {S,D,T,N,I} = __axes(A, StaticInt{I}())

@inline function __axes(A::AbstractStrideArray{S,D,T,N}, i::Integer) where {S,D,T,N}
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
@inline ArrayInterface.size(::AbstractStrideVector, ::StaticInt{N}) where {N} = One()
@inline ArrayInterface.size(A::AbstractStrideVector, ::StaticInt{1}) = length(A)
@inline ArrayInterface.size(A::AbstractStrideArray, ::StaticInt{N}) where {N} = size(A)[N]
@inline ArrayInterface.size(A::AbstractStrideArray, i::Integer) =
  type_stable_select(size(A), i)
@inline Base.size(A::AbstractStrideArray, i::Integer) = size(A, i)


# Base.IndexStyle(::Type{<:AbstractStrideArray}) = IndexCartesian()
# Base.IndexStyle(::Type{<:AbstractStrideVector{<:Any,<:Any,<:Any,1}}) = IndexLinear()
@generated function Base.IndexStyle(
  ::Type{<:AbstractStrideArray{S,D,T,N,C,B,R}},
) where {S,D,T,N,C,B,R}
  # if is column major || is a transposed contiguous vector
  if all(D) && (
    (isone(C) && R === ntuple(identity, Val(N))) ||
    (C === 2 && R === (2, 1) && S <: Tuple{One,Integer})
  )
    :(IndexLinear())
  else
    :(IndexCartesian())
  end
end

@inline LayoutPointers.preserve_buffer(A::PtrArray) = nothing


@generated function pload(p::Ptr{T}) where {T}
  if Base.allocatedinline(T)
    Expr(:block, Expr(:meta, :inline), :(unsafe_load(p)))
  else
    Expr(
      :block,
      Expr(:meta, :inline),
      :(ccall(
        :jl_value_ptr,
        Ref{$T},
        (Ptr{Cvoid},),
        unsafe_load(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, p)),
      )),
    )
  end
end
@generated function pstore!(p::Ptr{T}, v::T) where {T}
  if Base.allocatedinline(T)
    Expr(:block, Expr(:meta, :inline), :(unsafe_store!(p, v); return nothing))
  else
    Expr(
      :block,
      Expr(:meta, :inline),
      :(
        unsafe_store!(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, p), Base.pointer_from_objref(v)); return nothing
      ),
    )
  end
end
@inline pstore!(p::Ptr{T}, v) where {T} = pstore!(p, convert(T, v))

function rank2sortperm(R)
  map(R) do r
    sum(map(≥(r), R))
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
#   q = Expr(:block, Expr(:meta,:inline), :(p = pointer(ptr)), :(o = LayoutPointers.offsets(ptr)), :(x = strides(ptr)))
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
@generated function _offset_ptr(
  ptr::AbstractStridedPointer{T,N,C,B,R},
  i::Tuple{Vararg{Integer,NI}},
) where {T,N,C,B,R,NI}
  ptr_expr = :(pointer(ptr))
  T === Bit && (ptr_expr = :(Ptr{UInt8}($ptr_expr)))
  N == 0 && return Expr(:block, Expr(:meta, :inline), ptr_expr)
  if N ≠ NI
    if (N > NI) & (NI ≠ 1)
      throw(
        ArgumentError(
          "If the dimension of the array exceeds the dimension of the index, then the index should be linear/one dimensional.",
        ),
      )
    end
    # use only the first index. Supports, for example `x[i,1,1,1,1]` when `x` is a vector, or `A[i]` where `A` is an array with dim > 1.
    return Expr(
      :block,
      Expr(:meta, :inline),
      :($ptr_expr + (first(i) - 1) * $(static_sizeof(T))),
    )
  end
  sp = rank2sortperm(R)
  q = Expr(
    :block,
    Expr(:meta, :inline),
    :(p = $ptr_expr),
    :(o = offsets(ptr)),
    :(x = strides(ptr)),
  )
  for n ∈ 1:N
    j = findfirst(==(n), sp)::Int
    index = Expr(:call, getfield, :i, j, false)
    offst = Expr(:call, getfield, :o, j, false)
    strid = Expr(:call, getfield, :x, j, false)
    if T ≢ Bit
      push!(q.args, :(p += ($index - $offst) * $strid))
    else
      push!(q.args, :(p += (($index - $offst) * $strid) >>> 3))
    end
  end
  q
end
# @inline _offset_ptr(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,Zero}}, i::Tuple{Vararg{Integer,NI}}) where {T,N,C,B,R,NI,X} = __offset_ptr(ptr,i)
# @inline function _offset_ptr(ptr::AbstractStridedPointer{T,N,C,B,R}, i::Tuple{Vararg{Integer,NI}}) where {T,N,C,B,R,NI}
#   __offset_ptr(LayoutPointers.center(ptr), i)
# end


@inline function Base.getindex(
  A::BitPtrArray{S,D,N,C},
  i::Vararg{Integer,N},
) where {S,D,N,C}
  fi = getfield(i, C) - getfield(offsets(A), C)
  u = pload(_offset_ptr(stridedpointer(A), i))
  (u >>> (fi & 7)) % Bool
end
@inline function Base.getindex(A::BitPtrArray{S,D,N,C}, i::Integer) where {S,D,N,C}
  j = i - oneunit(i)
  u = pload(reinterpret(Ptr{UInt8}, pointer(A)) + (j >>> 3))
  (u >>> (j & 7)) % Bool
end
@inline function Base.setindex!(
  A::BitPtrArray{S,D,N,C},
  v::Bool,
  i::Vararg{Integer,N},
) where {S,D,N,C}
  fi = getfield(i, C) - getfield(offsets(A), C)
  p = _offset_ptr(stridedpointer(A), i)
  u = pload(p)
  sh = fi & 7
  u &= ~(0x01 << sh)
  u |= v << sh
  pstore!(p, u)
  return v
end
@inline function Base.setindex!(
  A::BitPtrArray{S,D,N,C},
  v::Bool,
  i::Integer,
) where {S,D,N,C}
  j = i - oneunit(i)
  p = Ptr{UInt8}(pointer(A)) + (j >>> 3)
  u = pload(p)
  sh = j & 7
  u &= ~(0x01 << sh) # 0 bit
  u |= v << sh
  pstore!(p, u)
  return v
end
# Base.@propagate_inbounds function Base.getindex(A::AbstractStrideArray, i::Vararg{Any,K}) where {K}
# end
Base.@propagate_inbounds Base.getindex(A::AbstractStrideVector, i::Integer, ::Integer) =
  getindex(A, i)
Base.@propagate_inbounds function Base.getindex(
  A::AbstractStrideArray,
  i::Vararg{Integer,K},
) where {K}
  b = preserve_buffer(A)
  GC.@preserve b begin
    PtrArray(A)[i...]
  end
end
Base.@propagate_inbounds function Base.setindex!(
  A::AbstractStrideArray,
  v,
  i::Vararg{Integer,K},
) where {K}
  b = preserve_buffer(A)
  GC.@preserve b begin
    PtrArray(A)[i...] = v
  end
end
boundscheck() = false
# Base.@propagate_inbounds Base.getindex(A::AbstractStrideVector, i::Int, j::Int) = A[i]
@inline function Base.getindex(A::PtrArray, i::Vararg{Integer})
  boundscheck() && @boundscheck checkbounds(A, i...)
  pload(_offset_ptr(stridedpointer(A), i))
end
# @inline function Base.getindex(A::AbstractStrideArray, i::Vararg{Integer,K}) where {K}
#   b = preserve_buffer(A)
#   P = PtrArray(A)
#   GC.@preserve b begin
#     @boundscheck checkbounds(P, i...)
#     pload(_offset_ptr(stridedpointer(P), i))
#   end
# end
@inline function Base.setindex!(A::PtrArray, v, i::Vararg{Integer,K}) where {K}
  boundscheck() && @boundscheck checkbounds(A, i...)
  pstore!(_offset_ptr(stridedpointer(A), i), v)
  v
end
# @inline function Base.setindex!(A::AbstractStrideArray, v, i::Vararg{Integer,K}) where {K}
#   b = preserve_buffer(A)
#   P = PtrArray(A)
#   GC.@preserve b begin
#     @boundscheck checkbounds(P, i...)
#     pstore!(_offset_ptr(stridedpointer(A), i), v)
#   end
#   v
# end
@inline function Base.getindex(A::PtrArray{S,D,T}, i::Integer) where {S,D,T}
  boundscheck() && @boundscheck checkbounds(A, i)
  pload(pointer(A) + (i - oneunit(i)) * static_sizeof(T))
end
# @inline function Base.getindex(A::AbstractStrideArray{S,D,T}, i::Integer) where {S,D,T}
#   b = preserve_buffer(A)
#   P = PtrArray(A)
#   GC.@preserve b begin
#     @boundscheck checkbounds(P, i)
#     pload(pointer(A) + (i-oneunit(i))*static_sizeof(T))
#   end
# end
@inline function Base.setindex!(A::PtrArray{S,D,T}, v, i::Integer) where {S,D,T}
  boundscheck() && @boundscheck checkbounds(A, i)
  pstore!(pointer(A) + (i - oneunit(i)) * static_sizeof(T), v)
  v
end
# @inline function Base.setindex!(A::AbstractStrideArray{S,D,T}, v, i::Integer) where {S,D,T}
#   b = preserve_buffer(A)
#   P = PtrArray(A)
#   GC.@preserve b begin
#     @boundscheck checkbounds(P, i)
#     pstore!(pointer(A) + (i-oneunit(i))*static_sizeof(T), v)
#   end
#   v
# end


@inline function Base.getindex(A::PtrVector{S,D,T}, i::Integer) where {S,D,T}
  boundscheck() && @boundscheck checkbounds(A, i)
  pload(pointer(A) + (i - ArrayInterface.offset1(A)) * only(LayoutPointers.bytestrides(A)))
end
# @inline function Base.getindex(A::AbstractStrideVector{S,D,T}, i::Integer) where {S,D,T}
#   b = preserve_buffer(A)
#   P = PtrArray(A)
#   GC.@preserve b begin
#     @boundscheck checkbounds(P, i)
#     pload(pointer(A) + (i-oneunit(i))*only(LayoutPointers.bytestrides(A)))
#   end
# end
@inline function Base.setindex!(A::PtrVector{S,D,T}, v, i::Integer) where {S,D,T}
  boundscheck() && @boundscheck checkbounds(A, i)
  pstore!(
    pointer(A) + (i - ArrayInterface.offset1(A)) * only(LayoutPointers.bytestrides(A)),
    v,
  )
  v
end
# @inline function Base.setindex!(A::AbstractStrideVector{S,D,T}, v, i::Integer) where {S,D,T}
#   b = preserve_buffer(A)
#   P = PtrArray(A)
#   GC.@preserve b begin
#     @boundscheck checkbounds(P, i)
#     pstore!(pointer(A) + (i-oneunit(i))*only(LayoutPointers.bytestrides(A)), v)
#   end
#   v
# end

@inline LayoutPointers.bytestrideindex(A::AbstractStrideArray{T}) where {T} =
  StrideIndex(stridedpointer(A))

_scale(::False, x, _, __) = x
@inline function _scale(::True, x, num, denom)
  cmp = Static.gt(num, denom)
  numerator = IfElse.ifelse(cmp, num, denom)
  denominator = IfElse.ifelse(cmp, denom, num)
  frac = numerator ÷ denominator
  IfElse.ifelse(cmp, x * frac, x ÷ frac)
end

@inline function Base.reinterpret(
  ::Type{Tnew},
  A::PtrArray{S,D,Told,N},
) where {Tnew,Told,S,D,N}
  sz = let szt_old = static_sizeof(Told), szt_new = static_sizeof(Tnew)
    map(
      _scale,
      contiguous_axis_indicator(A),
      size(A),
      ntuple(_ -> szt_old, Val(N)),
      ntuple(_ -> szt_new, Val(N)),
    )
  end
  sp = reinterpret(Tnew, stridedpointer(A))
  PtrArray(sp, sz, Val{D}())
end

@generated function Base.reinterpret(
  ::typeof(reshape),
  ::Type{Tnew},
  A::PtrArray{S,D,Told,N,C,B,R},
) where {S,D,Told,Tnew,N,C,B,R}
  sz_old = sizeof(Told)
  sz_new = sizeof(Tnew)
  Nnew = ifelse(sz_old == sz_new, N, ifelse(sz_old < sz_new, N - 1, N + 1))
  Bnew = ((B ≤ 0) | (sz_old == sz_new)) ? B : ((sz_old * B) ÷ sz_new)
  # sz_old < sz_new && push!(q.args, :(@assert size_A[$C] == $(sz_new ÷ sz_old)))
  if sz_old == sz_new
    size_expr = :size_A
    bx_expr = :bx
    offs_expr = :offs
    Rnew = R
    Cnew = C
    Dnew = D
    Nnew = N
  else
    @assert 1 ≤ C ≤ N
    size_expr = Expr(:tuple)
    bx_expr = Expr(:tuple)
    offs_expr = Expr(:tuple)
    Rnew = Expr(:tuple)
    Dnew = Expr(:tuple)
    for n ∈ 1:N
      sz_n = Expr(:call, GlobalRef(Core, :getfield), :size_A, n, false)
      bx_n = Expr(:call, GlobalRef(Core, :getfield), :bx, n, false)
      of_n = Expr(:call, GlobalRef(Core, :getfield), :offs, n, false)
      if n ≠ C
        push!(size_expr.args, sz_n)
        push!(bx_expr.args, bx_n)
        push!(offs_expr.args, of_n)
        push!(Dnew.args, D[n])
        r = R[n]
        r = if sz_old > sz_new
          r += r > C
        elseif sz_old < sz_new
          r -= r > C
        end
        push!(Rnew.args, r)
      elseif sz_old > sz_new
        si = :(StaticInt{$(sz_old ÷ sz_new)}())
        push!(size_expr.args, si, sz_n)
        push!(bx_expr.args, Expr(:call, :÷, bx_n, si), bx_n)
        push!(offs_expr.args, :(One()), of_n)
        push!(Dnew.args, true, D[n])
        push!(Rnew.args, 1, 1 + R[n])
        Cnew = C
      else
        # si = :(StaticInt{$(sz_new ÷ sz_old)}())
        # push!(size_expr.args, Expr(:call, :÷, sz_n, si))
        # push!(bx_expr.args, Expr(:call, :*, sz_n, si))
        R2ind = findfirst(==(2), R)
        Cnew = if R2ind === nothing
          0
        elseif D[R2ind]
          R2ind - 1
        else
          0
        end
      end
    end
  end
  quote
    $(Expr(:meta, :inline))
    sp = stridedpointer(A)
    bx = LayoutPointers.bytestrides(sp)
    size_A = size(A)
    offs = offsets(sp)
    si = StrideIndex{$Nnew,$Rnew,$Cnew}($bx_expr, $offs_expr)
    sp = stridedpointer(reinterpret(Ptr{$Tnew}, pointer(sp)), si, StaticInt{$Bnew}())
    PtrArray(sp, $size_expr, Val{$Dnew}())
  end
end
@inline Base.reinterpret(::Type{T}, A::AbstractStrideArray) where {T} =
  StrideArray(reinterpret(T, PtrArray(A)), preserve_buffer(A))
@inline Base.reinterpret(::typeof(reshape), ::Type{T}, A::AbstractStrideArray) where {T} =
  StrideArray(reinterpret(reshape, T, PtrArray(A)), preserve_buffer(A))
