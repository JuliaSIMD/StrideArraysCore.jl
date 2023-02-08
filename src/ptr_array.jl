
@generated function permtuple(x::Tuple, ::Val{R}) where {R}
  t = Expr(:tuple)
  for r in R
    push!(t.args, Expr(:call, getfield, :x, r))
  end
  Expr(:block, Expr(:meta, :inline), t)
end
@generated function invpermtuple(x::Tuple, ::Val{R}) where {R}
  t = Expr(:tuple)
  for i in eachindex(R)
    j = findfirst(==(i), R)::Int
    push!(t.args, Expr(:call, getfield, :x, j))
  end
  Expr(:block, Expr(:meta, :inline), t)
end

@inline _strides(::Tuple{}, ::Tuple{}) = ()
@inline _strides(::Tuple{}, ::Tuple{}, prev::Integer) = ()

struct StrideReset{T}
  x::T
end
@inline Base.:(*)(x::StrideReset, y::Union{Integer,StaticInt}) = x.x * y
@inline __scale(x::StrideReset, y::Union{Integer,StaticInt}) =
  StrideReset(x.x * y)
@inline __scale(x::Union{Integer,StaticInt}, y::Union{Integer,StaticInt}) =
  x * y
# three arg calcs div((x * y), z)
@inline __scale(
  x::StrideReset,
  y::Union{Integer,StaticInt},
  z::Union{Integer,StaticInt}
) = StrideReset(div(x.x * y, z))
@inline __scale(
  x::Union{Integer,StaticInt},
  y::Union{Integer,StaticInt},
  z::Union{Integer,StaticInt}
) = div(x * y, z)
# sizes M x N
# strides nothing x nothing -> static(1) x M
# strides L x nothing -> L x L*M
# strides nothing x K -> static(1) x K
# strides L x K -> L x L*K
@inline function _strides(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Integer,Vararg{Any,N}},
  prev::Integer
) where {N}
  next = prev * first(strides)
  (next, _strides(Base.tail(sizes), Base.tail(strides), next)...)
end
@inline function _strides(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Nothing,Vararg{Any,N}},
  prev::Integer
) where {N}
  next = prev * first(sizes)
  (next, _strides(Base.tail(sizes), Base.tail(strides), next)...)
end
@inline function _strides(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{StrideReset{T},Vararg{Any,N}},
  ::Integer
) where {N,T}
  next = getfield(first(strides), :x)
  (next, _strides(Base.tail(sizes), Base.tail(strides), next)...)
end

# entry point
@inline function _strides_nobit(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Integer,Vararg{Any,N}}
) where {N}
  prev = first(strides)
  (prev, _strides(Base.front(sizes), Base.tail(strides), prev)...)
end
@inline function _strides_nobit(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{StrideReset{T},Vararg{Any,N}}
) where {N,T}
  next = getfield(first(strides), :x)
  (next, _strides(Base.front(sizes), Base.tail(strides), next)...)
end
@inline function _strides_nobit(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Nothing,Vararg{Any,N}}
) where {N}
  prev = static(1)
  (prev, _strides(Base.front(sizes), Base.tail(strides), prev)...)
end

# entry point, guaranteed to have N > 0
@inline function _strides_bit(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Integer,Vararg{Any,N}}
) where {N}
  prev = (first(strides) + static(7)) & static(-8)
  (prev, _strides(Base.front(sizes), Base.tail(strides), prev)...)
end
@inline function _strides_bit(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{StrideReset{T},Vararg{Any,N}}
) where {N,T}
  next = (getfield(first(strides), :x) + static(7)) & static(-8)
  (next, _strides(Base.front(sizes), Base.tail(strides), next)...)
end
@inline function __strides_bit(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Integer,Vararg{Any,N}}
) where {N}
  next = (first(strides) + static(7)) & static(-8)
  (next, _strides(Base.tail(sizes), Base.tail(strides), next)...)
end
@inline function __strides_bit(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Nothing,Vararg{Any,N}}
) where {N}
  next = (first(sizes) + static(7)) & static(-8)
  (next, _strides(Base.tail(sizes), Base.tail(strides), next)...)
end
@inline function __strides_bit(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{StrideReset{T},Vararg{Any,N}}
) where {N,T}
  next = (getfield(first(strides), :x) + static(7)) & static(-8)
  (next, _strides(Base.tail(sizes), Base.tail(strides), next)...)
end
@inline function _strides_bit(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Nothing,Vararg{Any,N}}
) where {N}
  (static(1), __strides_bit(Base.front(sizes), Base.tail(strides))...)
end

# 4th arg is bit
@inline function _strides_entry(
  sizes,
  strides,
  ::Val{R},
  ::Val{false}
) where {R}
  VR = Val{R}()
  sx = _strides_nobit(invpermtuple(sizes, VR), invpermtuple(strides, VR))
  permtuple(sx, VR)
end
@inline _strides_entry(::Tuple{}, ::Tuple{}, ::Val{()}, ::Val{true}) = ()
@inline function _strides_entry(
  ::Tuple{S},
  ::Tuple{Nothing},
  ::Val{R},
  ::Val{true}
) where {S,R}
  (static(1),)
end
@inline function _strides_entry(
  ::Tuple{S},
  strides::Tuple{Integer},
  ::Val{R},
  ::Val{true}
) where {S,R}
  strides
end
@inline function _strides_entry(
  ::Tuple{S},
  strides::Tuple{StrideReset{T}},
  ::Val{R},
  ::Val{true}
) where {S,T,R}
  (getfield(only(strides), :x),)
end
@inline function _strides_entry(sizes, strides, ::Val{R}, ::Val{true}) where {R}
  VR = Val{R}()
  sx = _strides_bit(invpermtuple(sizes, VR), invpermtuple(strides, VR))
  permtuple(sx, VR)
end

@inline _dense_dims(::Tuple{}) = ()
@inline _dense_dims(x::Tuple{Nothing,Vararg{Any}}) =
  (True(), _dense_dims(Base.tail(x))...)
@inline _dense_dims(x::Tuple{Integer,Vararg{Any}}) =
  (False(), _dense_dims(Base.tail(x))...)
# @inline _dense_dims(x::Tuple, ::Val{R}) where {R} = _dense_dims(invpermtuple(x, Val{R}()))

abstract type AbstractStrideArray{
  T,
  N,
  R,
  S<:Tuple{Vararg{Integer,N}},
  X<:Tuple{Vararg{Union{Integer,Nothing,StrideReset},N}},
  O<:Tuple{Vararg{Integer,N}}
} <: DenseArray{T,N} end
abstract type AbstractPtrStrideArray{T,N,R,S,X,O} <:
              AbstractStrideArray{T,N,R,S,X,O} end
const AbstractStrideVector{T,R,S,X,O} = AbstractStrideArray{T,1,R,S,X,O}
const AbstractStrideMatrix{T,R,S,X,O} = AbstractStrideArray{T,2,R,S,X,O}

struct AbstractPtrArray{T,N,R,S,X,O,P} <: AbstractPtrStrideArray{T,N,R,S,X,O}
  ptr::Ptr{P}
  sizes::S
  strides::X
  offsets::O
  # function AbstractPtrArray{T,N,R,S,X,O,P}(ptr, sizes, strides, offsets) where {
  #   T,N,R,S,X,O,P
  #   }
  #   @assert T !== Bit
  #   new{T,N,R,S,X,O,P}(ptr, sizes, strides, offsets)
  # end
end
const PtrArray{T,N,R,S,X,O} = AbstractPtrArray{T,N,R,S,X,O,T}
const PtrArray0{T,N,R,S,X} = AbstractPtrArray{T,N,R,S,X,NTuple{N,Zero},T}
const PtrArray1{T,N,R,S,X} = AbstractPtrArray{T,N,R,S,X,NTuple{N,One},T}

const BitPtrArray{N,R,S,X,O} = AbstractPtrArray{Bool,N,R,S,X,O,Bit}
const BitPtrArray0{N,R,S,X} = AbstractPtrArray{Bool,N,R,S,X,NTuple{N,Zero},Bit}
const BitPtrArray1{N,R,S,X} = AbstractPtrArray{Bool,N,R,S,X,NTuple{N,One},Bit}

const AbstractPtrVector{T,R,S,X,O,P} = AbstractPtrArray{T,1,R,S,X,O,P}
const AbstractPtrMatrix{T,R,S,X,O,P} = AbstractPtrArray{T,2,R,S,X,O,P}

const PtrVector{T,R,S,X,O} = AbstractPtrArray{T,1,R,S,X,O,T}
const PtrMatrix{T,R,S,X,O} = AbstractPtrArray{T,2,R,S,X,O,T}
const PtrVector0{T,R,S,X} = AbstractPtrArray{T,1,R,S,X,NTuple{1,Zero},T}
const PtrVector1{T,R,S,X} = AbstractPtrArray{T,1,R,S,X,NTuple{1,One},T}
const PtrMatrix0{T,R,S,X} = AbstractPtrArray{T,2,R,S,X,NTuple{2,Zero},T}
const PtrMatrix1{T,R,S,X} = AbstractPtrArray{T,2,R,S,X,NTuple{2,One},T}

const BitPtrVector{R,S,X,O} = AbstractPtrArray{Bool,1,R,S,X,O,Bit}
const BitPtrMatrix{R,S,X,O} = AbstractPtrArray{Bool,2,R,S,X,O,Bit}
const BitPtrVector0{R,S,X} = AbstractPtrArray{Bool,1,R,S,X,NTuple{1,Zero},Bit}
const BitPtrVector1{R,S,X} = AbstractPtrArray{Bool,1,R,S,X,NTuple{1,One},Bit}
const BitPtrMatrix0{R,S,X} = AbstractPtrArray{Bool,2,R,S,X,NTuple{2,Zero},Bit}
const BitPtrMatrix1{R,S,X} = AbstractPtrArray{Bool,2,R,S,X,NTuple{2,One},Bit}

@inline valisbit(::AbstractPtrArray{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,Bit}) =
  Val(true)
@inline valisbit(
  ::AbstractPtrArray{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}
) = Val(false)

# function PtrArray(
#   ptr::Ptr{T}, sizes::S, strides::X, offsets::O, ::Val{R}
# ) where {T,N,R,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Integer,N}}}
#   PtrArray{T,N,R,S,X,O}(ptr, sizes, strides, offsets)
# end

@inline function AbstractPtrArray(
  p::Ptr{T},
  sz::S,
  sx::X,
  so::O,
  ::Val{R}
) where {
  N,
  T,
  R,
  S<:Tuple{Vararg{Integer,N}},
  X<:Tuple{Vararg{Any,N}},
  O<:Tuple{Vararg{Integer,N}}
}
  AbstractPtrArray{T,N,R,S,X,O,T}(p, sz, sx, so)
end
@inline function AbstractPtrArray(
  p::Ptr{Bit},
  sz::S,
  sx::X,
  so::O,
  ::Val{R}
) where {
  N,
  R,
  S<:Tuple{Vararg{Integer,N}},
  X<:Tuple{Vararg{Any,N}},
  O<:Tuple{Vararg{Integer,N}}
}
  AbstractPtrArray{Bool,N,R,S,X,O,Bit}(p, sz, sx, so)
end

@inline function AbstractPtrArray(
  p::Ptr{T},
  sz::S,
  sx::X,
  so::O
) where {
  N,
  T,
  S<:Tuple{Vararg{Integer,N}},
  X<:Tuple{Vararg{Any,N}},
  O<:Tuple{Vararg{Integer,N}}
}
  AbstractPtrArray{T,N,ntuple(identity, Val(N)),S,X,O,T}(p, sz, sx, so)
end
@inline function AbstractPtrArray(
  p::Ptr{Bit},
  sz::S,
  sx::X,
  so::O
) where {
  N,
  S<:Tuple{Vararg{Integer,N}},
  X<:Tuple{Vararg{Any,N}},
  O<:Tuple{Vararg{Integer,N}}
}
  AbstractPtrArray{Bool,N,ntuple(identity, Val(N)),S,X,O,Bit}(p, sz, sx, so)
end

@inline function PtrArray(
  p::Ptr{T},
  sz::S,
  ::Val{R}
) where {T,N,S<:Tuple{Vararg{Integer,N}},R}
  sx = ntuple(Returns(nothing), Val(N))
  o = ntuple(Returns(static(1)), Val(N))
  AbstractPtrArray(p, sz, sx, o, Val(R))
end
@inline function PtrArray0(
  p::Ptr{T},
  sz::S,
  ::Val{R}
) where {T,N,S<:Tuple{Vararg{Integer,N}},R}
  sx = ntuple(Returns(nothing), Val(N))
  o = ntuple(Returns(static(0)), Val(N))
  AbstractPtrArray(p, sz, sx, o, Val(R))
end
@inline function PtrArray(
  p::Ptr{T},
  sz::S,
  sx::X,
  ::Val{R}
) where {T,N,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},R}
  o = ntuple(Returns(static(1)), Val(N))
  AbstractPtrArray(p, sz, sx, o, Val(R))
end
@inline function PtrArray0(
  p::Ptr{T},
  sz::S,
  sx::X,
  ::Val{R}
) where {T,N,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},R}
  o = ntuple(Returns(static(0)), Val(N))
  AbstractPtrArray(p, sz, sx, o, Val(R))
end
@inline function PtrArray(
  p::Ptr{T},
  sz::S
) where {T,N,S<:Tuple{Vararg{Integer,N}}}
  sx = ntuple(Returns(nothing), Val(N))
  o = ntuple(Returns(static(1)), Val(N))
  R = ntuple(identity, Val(N))
  AbstractPtrArray(p, sz, sx, o, Val(R))
end
@inline function PtrArray0(
  p::Ptr{T},
  sz::S
) where {T,N,S<:Tuple{Vararg{Integer,N}}}
  sx = ntuple(Returns(nothing), Val(N))
  o = ntuple(Returns(static(0)), Val(N))
  R = ntuple(identity, Val(N))
  AbstractPtrArray(p, sz, sx, o, Val(R))
end
@inline function PtrArray(
  p::Ptr{T},
  sz::S,
  sx::X
) where {T,N,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}}}
  o = ntuple(Returns(static(1)), Val(N))
  R = ntuple(identity, Val(N))
  AbstractPtrArray(p, sz, sx, o, Val(R))
end
@inline function PtrArray0(
  p::Ptr{T},
  sz::S,
  sx::X
) where {T,N,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}}}
  o = ntuple(Returns(static(0)), Val(N))
  R = ntuple(identity, Val(N))
  AbstractPtrArray(p, sz, sx, o, Val(R))
end

@generated function _nondense_strides(
  strides::Tuple{Vararg{Integer}},
  ::Val{F},
  ::Val{D}
) where {F,D}
  t = Expr(:tuple)
  for i in eachindex(D)
    if D[i]
      push!(t.args, nothing)
    else
      x = Expr(:call, getfield, :strides, i)
      if F
        x = Expr(:call, >>>, x, static(3))
      end
      push!(t.args, Expr(:call, StrideReset, x))
    end
  end
  Expr(:block, Expr(:meta, :inline), t)
end

@inline function LayoutPointers.stridedpointer(A::BitPtrArray)
  stridedpointer(getfield(A, :ptr), StrideIndex(A))
end

@inline function PtrArray(
  ptr::StridedPointer{T,N,<:Any,0,R,<:Any,O},
  sz::S,
  ::Val{D}
) where {T,N,R,O,D,S}
  sx = _nondense_strides(strides(ptr), Val(true), Val{D}())
  X = typeof(sx)
  PtrArray{T,N,R,S,X,O}(pointer(ptr), sz, sx, offsets(ptr))
end
@inline function PtrArray(
  ptr::StridedBitPointer{N,<:Any,0,R,<:Any,O},
  sz::S,
  ::Val{D}
) where {N,R,O,D,S}
  sx = _nondense_strides(strides(ptr), Val(false), Val{D}())
  X = typeof(sx)
  BitPtrArray{N,R,S,X,O}(pointer(ptr), sz, sx, offsets(ptr))
end
@inline PtrArray(A::BitArray{N}) where {N} =
  PtrArray(stridedpointer(A), size(A), Val(ntuple(Returns(true), Val(N))))

@inline _sparse_strides(::Tuple{}, ::Tuple{}) = ()
@inline function _sparse_strides(
  dd::Tuple{True,Vararg{Any,N}},
  sx::Tuple{Integer,Vararg{Integer,N}}
) where {N}
  (nothing, _sparse_strides(Base.tail(dd), Base.tail(sx))...)
end
@inline function _sparse_strides(
  dd::Tuple{False,Vararg{Any,N}},
  sx::Tuple{Integer,Vararg{Integer,N}}
) where {N}
  (StrideReset(first(sx)), _sparse_strides(Base.tail(dd), Base.tail(sx))...)
end

@inline function PtrArray(
  p::Ptr{T},
  sz::S,
  sx::X,
  offsets::O,
  ::Val{R}
) where {
  T,
  N,
  R,
  S<:Tuple{Vararg{Integer,N}},
  X<:Tuple{Vararg{Any,N}},
  O<:Tuple{Vararg{Integer,N}}
}
  AbstractPtrArray{T,N,R,S,X,O,T}(p, sz, sx, offsets)
end
@inline function PtrArray(
  p::Ptr{Bit},
  sz::S,
  sx::X,
  offsets::O,
  ::Val{R}
) where {
  N,
  R,
  S<:Tuple{Vararg{Integer,N}},
  X<:Tuple{Vararg{Any,N}},
  O<:Tuple{Vararg{Integer,N}}
}
  AbstractPtrArray{Bool,N,R,S,X,O,Bit}(p, sz, sx, offsets)
end

@generated _compact_rank(::Val{R}) where {R} = Val(map(r -> sum(<=(r), R), R))

@inline function PtrArray(A::AbstractArray{T,N}) where {T,N}
  p = LayoutPointers.memory_reference(A)[1]
  sz = size(A)
  sx = _sparse_strides(dense_dims(A), strides(A))
  R = map(Int, stride_rank(A))
  PtrArray(p, sz, sx, offsets(A), _compact_rank(Val(R)))
end

@inline Base.pointer(A::AbstractPtrStrideArray) = getfield(A, :ptr)

@inline Base.unsafe_convert(::Type{Ptr{T}}, A::AbstractStrideArray) where {T} =
  Base.unsafe_convert(Ptr{T}, pointer(A))
@inline Base.elsize(::AbstractStrideArray{T}) where {T} = sizeof(T)

@inline ArrayInterface.size(A::AbstractPtrStrideArray) = getfield(A, :sizes)
@inline function ArrayInterface.strides(
  A::AbstractPtrStrideArray{<:Any,<:Any,R}
) where {R}
  _strides_entry(size(A), getfield(A, :strides), Val{R}(), valisbit(A))
end
ArrayInterface.device(::AbstractStrideArray) = ArrayInterface.CPUPointer()

@generated function ArrayInterface.contiguous_axis(
  ::Type{<:AbstractStrideArray{<:Any,<:Any,R,<:Any,X}}
) where {R,X}
  i = findfirst(isone, R)
  C = i === nothing ? -1 : (X.parameters[i] === Nothing ? i : -1)
  StaticInt{C}()
end
ArrayInterface.contiguous_batch_size(::Type{<:AbstractStrideArray}) =
  StaticInt{0}()

ArrayInterface.known_size(
  ::Type{<:AbstractStrideArray{<:Any,<:Any,<:Any,S}}
) where {S} = Static.known(S)

@generated function ArrayInterface.stride_rank(
  ::Type{<:AbstractStrideArray{<:Any,<:Any,R}}
) where {R}
  t = Expr(:tuple)
  for r ∈ R
    push!(t.args, StaticInt{r}())
  end
  t
end
@generated function ArrayInterface.dense_dims(
  ::Type{<:AbstractStrideArray{<:Any,<:Any,<:Any,<:Any,X}}
) where {X}
  t = Expr(:tuple)
  for i in eachindex(X.parameters)
    if X.parameters[i] === Nothing
      push!(t.args, True())
    else
      push!(t.args, False())
    end
  end
  t
end

function onetupleexpr(N::Int)
  t = Expr(:tuple)
  for _ = 1:N
    push!(t.args, One())
  end
  Expr(:block, Expr(:meta, :inline), t)
end
@generated onetuple(::Val{N}) where {N} = onetupleexpr(N)

@inline function ptrarray0(
  p::Ptr{T},
  s::Tuple{Vararg{Union{Integer,StaticInt},N}},
  x::Tuple{Vararg{Union{Integer,StaticInt},N}},
  ::Val{D}
) where {T,N,D}
  PtrArray0(p, s, _nondense_strides(x, Val(false), Val{D}()))
end
@inline function PtrArray(
  p::Ptr{T},
  s::Tuple{Vararg{Union{Integer,StaticInt},N}},
  x::Tuple{Vararg{Union{Integer,StaticInt},N}},
  ::Val{D}
) where {T,N,D}
  PtrArray(p, s, _nondense_strides(x, Val(false), Val{D}()))
end

@inline sparse_strides(A::AbstractPtrStrideArray) = getfield(A, :strides)

@inline function LayoutPointers.zero_offsets(
  A::AbstractPtrStrideArray{<:Any,N,R}
) where {N,R}
  PtrArray(
    pointer(A),
    size(A),
    sparse_strides(A),
    ntuple(Returns(static(0)), Val(N)),
    Val{R}()
  )
end

intlog2(N::I) where {I<:Integer} = (8sizeof(I) - one(I) - leading_zeros(N)) % I
intlog2(::Type{T}) where {T} = intlog2(static_sizeof(T))
@generated intlog2(::StaticInt{N}) where {N} =
  Expr(:call, Expr(:curly, :StaticInt, intlog2(N)))

@inline Base.size(A::AbstractStrideArray) = map(Int, size(A))
@inline Base.strides(A::AbstractStrideArray) = map(Int, strides(A))
@inline function Base.stride(A::AbstractStrideArray, i::Int)
  x = Base.strides(A)
  @assert i > 0
  i <= length(x) ? @inbounds(x[i]) : last(x) * Int(last(size(A)))
end
@inline Base.stride(A::AbstractStrideArray, ::StaticInt{N}) where {N} =
  Base.stride(A, N::Int)
@generated _oneto(x) = Expr(:new, Base.OneTo{Int}, :(x % Int))

@inline create_axis(s, ::Zero) = CloseOpen(s)
@inline create_axis(s, ::One) = _oneto(unsigned(s))
@inline create_axis(::StaticInt{N}, ::One) where {N} = One():StaticInt{N}()
@inline create_axis(s, o) = CloseOpen(o, s + o)

@inline ArrayInterface.axes(A::AbstractPtrArray) =
  map(create_axis, size(A), offsets(A))
@inline ArrayInterface.size(A::AbstractStrideArray) = size(PtrArray(A))
@inline ArrayInterface.axes(A::AbstractStrideArray) = axes(PtrArray(A))
@inline Base.axes(A::AbstractStrideArray) = axes(A)
@inline Base.axes(A::AbstractStrideArray, d::StaticInt) = axes(A, d)

@generated function ArrayInterface.axes_types(
  ::Type{<:AbstractStrideArray{T,N,R,S,X,O}}
) where {T,N,R,S,X,O}
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
        push!(t.args, Base.OneTo{Int})
      else
        @assert si isa Int
        push!(
          t.args,
          ArrayInterface.OptionallyStaticUnitRange{StaticInt{1},StaticInt{si}}
        )
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

@inline ArrayInterface.offsets(A::AbstractPtrArray) = getfield(A, :offsets)
@inline ArrayInterface.static_length(A::AbstractStrideArray) =
  Static.reduce_tup(*, size(A))

# type stable, because index known at compile time
@inline type_stable_select(t::NTuple, ::StaticInt{N}) where {N} = getfield(t, N)
@inline type_stable_select(t::Tuple, ::StaticInt{N}) where {N} = getfield(t, N)
# type stable, because tuple is homogenous
@inline type_stable_select(t::NTuple, i::Integer) = getfield(t, i)
# make the tuple homogenous before indexing
@inline type_stable_select(t::Tuple, i::Integer) = getfield(map(Int, t), i)

@inline ArrayInterface._axes(A::AbstractStrideArray, i::Integer) = __axes(A, i)
@inline ArrayInterface._axes(A::AbstractStrideArray, i::Int) = __axes(A, i)
@inline ArrayInterface._axes(A::AbstractStrideArray, ::StaticInt{I}) where {I} =
  __axes(A, StaticInt{I}())

@inline function __axes(
  A::AbstractStrideArray{T,N},
  i::Union{Integer,StaticInt}
) where {T,N}
  if i ≤ N
    o = type_stable_select(offsets(A), i)
    s = type_stable_select(size(A), i)
    return create_axis(s, o)
  else
    return One():One()
  end
end
@inline Base.axes(A::AbstractStrideArray, i::Integer) = axes(A, i)

@inline function ArrayInterface.size(A::AbstractStrideVector, i::Int)
  d = Int(length(A))
  ifelse(isone(i), d, one(d))
end
@inline ArrayInterface.size(::AbstractStrideVector, ::StaticInt{N}) where {N} =
  One()
@inline ArrayInterface.size(A::AbstractStrideVector, ::StaticInt{1}) = length(A)
@inline ArrayInterface.size(A::AbstractStrideArray, ::StaticInt{N}) where {N} =
  size(A)[N]
@inline ArrayInterface.size(A::AbstractStrideArray, i::Int) =
  type_stable_select(size(A), i)
@inline Base.size(A::AbstractStrideArray, i::Union{Integer,StaticInt})::Int =
  size(A, i)

# Base.IndexStyle(::Type{<:AbstractStrideArray}) = IndexCartesian()
# Base.IndexStyle(::Type{<:AbstractStrideVector{<:Any,<:Any,<:Any,1}}) = IndexLinear()
@generated function Base.IndexStyle(
  ::Type{A}
) where {T,N,R,S,X,A<:AbstractStrideArray{T,N,R,S,X}}
  # if is column major || is a transposed contiguous vector
  if X === NTuple{N,Nothing} && (
    (R === ntuple(identity, Val(N))) ||
    (R === (2, 1) && S <: Tuple{One,Integer})
  )
    :(IndexLinear())
  else
    :(IndexCartesian())
  end
end

@inline ManualMemory.preserve_buffer(::PtrArray) = nothing

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
        unsafe_load(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, p))
      ))
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
        unsafe_store!(
          Base.unsafe_convert(Ptr{Ptr{Cvoid}}, p),
          Base.pointer_from_objref(v)
        );
        return nothing
      )
    )
  end
end
@inline pstore!(p::Ptr{T}, v) where {T} = pstore!(p, convert(T, v))

rank2sortperm(R) =
  map(R) do r
    sum(map(≥(r), R))
  end

@generated function _offset_ptr(
  ptr::AbstractStridedPointer{T,N,C,B,R},
  i::Tuple{Vararg{Union{Integer,StaticInt,AbstractRange,Colon},NI}}
) where {T,N,C,B,R,NI}
  ptr_expr = :(pointer(ptr))
  N == 0 && return Expr(:block, Expr(:meta, :inline), ptr_expr)
  if N ≠ NI
    if (N > NI) & (NI ≠ 1)
      throw(
        ArgumentError(
          "If the dimension of the array exceeds the dimension of the index, then the index should be linear/one dimensional."
        )
      )
    end
    # use only the first index. Supports, for example `x[i,1,1,1,1]` when `x` is a vector, or `A[i]` where `A` is an array with dim > 1.
    i.parameters[1] === Colon &&
      return Expr(:block, Expr(:meta, :inline), ptr_expr)
    iexpr = :(first(i))
    if i.parameters[1] <: AbstractRange
      iexpr = :(first($iexpr))
    end
    return Expr(
      :block,
      Expr(:meta, :inline),
      :($ptr_expr + ($iexpr - (ptr).si.offsets[1]) * $(static_sizeof(T)))
    )
  end
  sp = rank2sortperm(R)
  q = Expr(
    :block,
    Expr(:meta, :inline),
    :(p = $ptr_expr),
    :(o = offsets(ptr)),
    :(x = strides(ptr))
  )
  for n ∈ 1:N
    j = findfirst(==(n), sp)::Int
    ityp = i.parameters[j]
    ityp === Colon && continue
    index = Expr(:call, getfield, :i, j)
    if ityp <: AbstractRange
      index = :(first($index))
    end
    offst = Expr(:call, getfield, :o, j)
    strid = Expr(:call, getfield, :x, j)
    if T ≢ Bit
      push!(q.args, :(p += ($index - $offst) * $strid))
    else
      push!(q.args, :(p += (($index - $offst) * $strid) >>> 3))
    end
  end
  push!(q.args, :(p))
  q
end

@inline function Base.getindex(
  A::BitPtrArray{N},
  i::Vararg{Union{Integer,StaticInt},N}
) where {N}
  C = Int(ArrayInterface.contiguous_axis(A))::Int
  fi = getfield(i, C) - getfield(offsets(A), C)
  u = pload(Ptr{UInt8}(_offset_ptr(stridedpointer(A), i)))
  (u >>> (fi & 7)) % Bool
end
@inline function Base.getindex(A::BitPtrArray, i::Union{Integer,StaticInt})
  j = i - oneunit(i)
  u = pload(Ptr{UInt8}(pointer(A)) + (j >>> 3))
  (u >>> (j & 7)) % Bool
end
@inline function Base.setindex!(
  A::BitPtrArray{N},
  v::Bool,
  i::Vararg{Union{Integer,StaticInt},N}
) where {N}
  C = Int(ArrayInterface.contiguous_axis(A))::Int
  fi = getfield(i, C) - getfield(offsets(A), C)
  p = Ptr{UInt8}(_offset_ptr(stridedpointer(A), i))
  u = pload(p)
  sh = fi & 7
  u &= ~(0x01 << sh)
  u |= v << sh
  pstore!(p, u)
  return v
end
@inline function Base.setindex!(
  A::BitPtrArray,
  v::Bool,
  i::Union{Integer,StaticInt}
)
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
Base.@propagate_inbounds Base.getindex(
  A::AbstractStrideVector,
  i::Integer,
  ::Colon
) = view(A, i, :)
Base.@propagate_inbounds Base.getindex(
  A::AbstractStrideVector,
  ::Colon,
  ::Integer
) = A

Base.@propagate_inbounds Base.getindex(
  A::AbstractStrideVector,
  i::Integer,
  ::Integer
) = getindex(A, i)
Base.@propagate_inbounds function Base.getindex(
  A::AbstractStrideArray,
  i::Vararg{Union{Integer,StaticInt},K}
) where {K}
  b = preserve_buffer(A)
  GC.@preserve b begin
    PtrArray(A)[i...]
  end
end
Base.@propagate_inbounds function Base.getindex(
  A::AbstractStrideArray,
  i::Vararg{Union{Integer,StaticInt,Colon,AbstractRange},K}
) where {K}
  view(A, i...)
end
Base.@propagate_inbounds function Base.setindex!(
  A::AbstractStrideArray,
  v,
  i::Vararg{Union{Integer,StaticInt},K}
) where {K}
  b = preserve_buffer(A)
  GC.@preserve b begin
    PtrArray(A)[i...] = v
  end
end
boundscheck() = false
@inline function Base.getindex(A::PtrArray, i::Vararg{Integer})
  boundscheck() && @boundscheck checkbounds(A, i...)
  pload(_offset_ptr(stridedpointer(A), i))
end
@inline function Base.setindex!(A::PtrArray, v, i::Vararg{Integer,K}) where {K}
  boundscheck() && @boundscheck checkbounds(A, i...)
  pstore!(_offset_ptr(stridedpointer(A), i), v)
  v
end
@inline function Base.getindex(A::PtrArray{T}, i::Integer) where {T}
  boundscheck() && @boundscheck checkbounds(A, i)
  pload(pointer(A) + (i - oneunit(i)) * static_sizeof(T))
end
@inline function Base.setindex!(A::PtrArray{T}, v, i::Integer) where {T}
  boundscheck() && @boundscheck checkbounds(A, i)
  pstore!(pointer(A) + (i - oneunit(i)) * static_sizeof(T), v)
  v
end
@inline function Base.getindex(A::PtrVector{T}, i::Integer) where {T}
  boundscheck() && @boundscheck checkbounds(A, i)
  pload(
    pointer(A) +
    (i - ArrayInterface.offset1(A)) * only(LayoutPointers.bytestrides(A))
  )
end
@inline function Base.setindex!(A::PtrVector{T}, v, i::Integer) where {T}
  boundscheck() && @boundscheck checkbounds(A, i)
  pstore!(
    pointer(A) +
    (i - ArrayInterface.offset1(A)) * only(LayoutPointers.bytestrides(A)),
    v
  )
  v
end

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
  A::PtrArray{Told,N}
) where {Tnew,Told,N}
  szt_old = static_sizeof(Told)
  szt_new = static_sizeof(Tnew)
  sz_old = size(A)
  sz1_new =
    _scale(first(contiguous_axis_indicator(A)), first(sz_old), szt_old, szt_new)
  sz_new = (sz1_new, Base.tail(sz_old)...)
  sp = reinterpret(Tnew, stridedpointer(A))
  PtrArray(sp, sz_new, val_dense_dims(A))
end

@generated function Base.reinterpret(
  ::typeof(reshape),
  ::Type{Tnew},
  A::PtrArray{Told,N,R,S,X,O}
) where {Told,Tnew,N,S,R,X,O}
  sz_old::Int = sizeof(Told)::Int
  sz_new::Int = sizeof(Tnew)::Int
  C = findfirst(==(1), R)::Int
  # sz_old < sz_new && push!(q.args, :(@assert size_A[$C] == $(sz_new ÷ sz_old)))
  if sz_old == sz_new
    size_expr = :s
    stride_expr = :x
    offs_expr = :o
    Rnew = R
  else
    size_expr = Expr(:tuple)
    stride_expr = Expr(:tuple)
    offs_expr = Expr(:tuple)
    Rnew = Expr(:tuple)
    if sz_old >= sz_new
      known_offsets = known(O)
      first_offset =
        if all(Base.Fix2(isa, Int), known_offsets) &&
           all(==(first(known_offsets)), known_offsets)
          first(known_offsets)
        else
          1
        end
      push!(offs_expr.args, static(first_offset))
      push!(Rnew.args, 1)
      push!(size_expr.args, :(StaticInt{$(sz_old ÷ sz_new)}()))
      push!(stride_expr.args, nothing)
    end
    for n ∈ 1:N
      sz_n = Expr(:call, getfield, :s, n)
      sx_n = Expr(:call, getfield, :x, n)
      of_n = Expr(:call, getfield, :o, n)
      if (sz_old != sz_new) && (X.parameters[n] ≢ Nothing)
        sx_n = Expr(:call, __scale, sx_n, static(sz_old), static(sz_new))
      end
      if n ≠ C
        push!(size_expr.args, sz_n)
        push!(stride_expr.args, sx_n)
        push!(offs_expr.args, of_n)
        r = R[n]
        r = if sz_old > sz_new
          r += 1#r > C
        elseif sz_old < sz_new
          r -= r > C
        end
        push!(Rnew.args, r)
      elseif sz_old > sz_new
        # add an axis
        push!(size_expr.args, sz_n)
        push!(stride_expr.args, sx_n)
        push!(offs_expr.args, of_n)
        push!(Rnew.args, 1 + R[n])
      end
    end
  end
  quote
    $(Expr(:meta, :inline))
    p = pointer(A)
    s = getfield(A, :sizes)
    x = getfield(A, :strides)
    o = getfield(A, :offsets)
    PtrArray(Ptr{$Tnew}(p), $size_expr, $stride_expr, $offs_expr, Val{$Rnew}())
  end
end

Base.LinearIndices(x::AbstractStrideVector) = axes(x, static(1))
