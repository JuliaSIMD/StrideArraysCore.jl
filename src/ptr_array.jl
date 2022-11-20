

@generated function permtuple(x::Tuple, ::Val{R}) where {R}
  t = Expr(:tuple)
  for r = R
    push!(t.args, Expr(:call, getfield, :x, r))
  end
  Expr(:block,Expr(:meta,:inline), t)
end
@generated function invpermtuple(x::Tuple, ::Val{R}) where {R}
  t = Expr(:tuple)
  for i = eachindex(R)
    j = findfirst(==(i), R)::Int
    push!(t.args, Expr(:call, getfield, :x, j))
  end
  Expr(:block,Expr(:meta,:inline), t)
end

@inline _strides(::Tuple{}, ::Tuple{}) = ()
@inline _strides(::Tuple{}, ::Tuple{}, prev) = ()

struct StrideReset{T}; x::T; end

# sizes M x N
# strides nothing x nothing -> static(1) x M
# strides L x nothing -> L x L*M
# strides nothing x K -> static(1) x K
# strides L x K -> L x L*K
@inline function _strides(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Integer,Vararg{Any,N}}, prev::Integer
) where {N}
  next = prev*first(strides)
  (next, _strides(Base.tail(sizes), Base.tail(strides), next)...)
end
@inline function _strides(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Nothing,Vararg{Any,N}}, prev::Integer
) where {N}
  next = prev*first(sizes)
  (next, _strides(Base.tail(sizes), Base.tail(strides), next)...)
end
@inline function _strides(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{StrideReset{T},Vararg{Any,N}}, ::Integer
) where {N,T}
  next = getfield(first(strides), :x)
  (next, _strides(Base.tail(sizes), Base.tail(strides), next)...)
end

@inline function _strides(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Integer,Vararg{Any,N}}
) where {N}
  prev = first(strides)
  (prev, _strides(Base.front(sizes), Base.tail(strides), prev)...)
end
@inline function _strides(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{StrideReset{T},Vararg{Any,N}}
) where {N,T}
  next = getfield(first(strides), :x)
  (next, _strides(Base.tail(sizes), Base.tail(strides), next)...)
end
@inline function _strides(
  sizes::Tuple{Integer,Vararg{Integer,N}},
  strides::Tuple{Nothing,Vararg{Any,N}}
) where {N}
  prev = static(1)
  (prev, _strides(Base.front(sizes), Base.tail(strides), prev)...)
end

@inline function _strides(sizes, strides, ::Val{R}) where {R}
  VR = Val{R}()
  sx = _strides(invpermtuple(sizes, VR), invpermtuple(strides, VR))
  permtuple(sx, VR)
end


@inline _dense_dims(::Tuple{}) = ()
@inline _dense_dims(x::Tuple{Nothing,Vararg{Any}}) = (True(), _dense_dims(Base.tail(x))...)
@inline _dense_dims(x::Tuple{Integer,Vararg{Any}}) = (False(), _dense_dims(Base.tail(x))...)
# @inline _dense_dims(x::Tuple, ::Val{R}) where {R} = _dense_dims(invpermtuple(x, Val{R}()))

abstract type AbstractStrideArray{T,N,R,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Union{Integer,Nothing},N}},O<:Tuple{Vararg{Integer,N}}} <: DenseArray{T,N} end
abstract type AbstractPtrStrideArray{T,N,R,S,X,O} <:
              AbstractStrideArray{T,N,R,S,X,O} end
const AbstractStrideVector{T,R,S,X,O} = AbstractStrideArray{T,1,R,S,X,O}
const AbstractStrideMatrix{T,R,S,X,O} = AbstractStrideArray{T,2,R,S,X,O}

struct AbstractPtrArray{T,N,R,S,X,O,P} <: AbstractPtrStrideArray{T,N,R,S,X,O}
  ptr::Ptr{P}
  sizes::S
  strides::X
  offsets::O
end
const PtrArray{T,N,R,S,X,O} = AbstractPtrArray{T,N,R,S,X,O,T}
const PtrArray0{T,N,R,S,X} = AbstractPtrArray{T,N,R,S,X,NTuple{N,Zero},T}
const PtrArray1{T,N,R,S,X} = AbstractPtrArray{T,N,R,S,X,NTuple{N,One},T}
const BitPtrArray{N,R,S,X,O} = AbstractPtrArray{Bool,N,R,S,X,O,Bit}
const BitPtrArray0{N,R,S,X} = AbstractPtrArray{Bool,N,R,S,X,NTuple{N,Zero},Bit}
const BitPtrArray1{N,R,S,X} = AbstractPtrArray{Bool,N,R,S,X,NTuple{N,One},Bit}

# function PtrArray(
#   ptr::Ptr{T}, sizes::S, strides::X, offsets::O, ::Val{R}
# ) where {T,N,R,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Integer,N}}}
#   PtrArray{T,N,R,S,X,O}(ptr, sizes, strides, offsets)
# end


@inline function AbstractPtrArray(
  p::Ptr{T}, sz::S, sx::X, so::O, ::Val{R}
) where {N,T,R,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Integer,N}}}
  AbstractPtrArray{T,N,R,S,X,O,T}(p, sz, sx, so)
end
@inline function AbstractPtrArray(
  p::Ptr{Bit}, sz::S, sx::X, so::O, ::Val{R}
) where {N,R,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Integer,N}}}
  AbstractPtrArray{Bool,N,R,S,X,O,Bit}(p, sz, sx, so)
end

@inline function AbstractPtrArray(
  p::Ptr{T}, sz::S, sx::X, so::O
) where {N,T,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Integer,N}}}
  AbstractPtrArray{T,N,ntuple(identity,Val(N)),S,X,O,T}(p, sz, sx, so)
end
@inline function AbstractPtrArray(
  p::Ptr{Bit}, sz::S, sx::X, so::O
) where {N,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Integer,N}}}
  AbstractPtrArray{Bool,N,ntuple(identity,Val(N)),S,X,O,Bit}(p, sz, sx, so)
end

@inline function PtrArray(p::Ptr{T}, sz::S, ::Val{R}) where {T,N,S<:Tuple{Vararg{Integer,N}},R}
  sx = ntuple(Returns(nothing), Val(N))
  o = ntuple(Returns(static(1)), Val(N))
  PtrArray{T,N,R,S,NTuple{N,Nothing},NTuple{N,StaticInt{1}}}(p, sz, sx, o)
end
@inline function PtrArray0(p::Ptr{T}, sz::S, ::Val{R}) where {T,N,S<:Tuple{Vararg{Integer,N}},R}
  sx = ntuple(Returns(nothing), Val(N))
  o = ntuple(Returns(static(0)), Val(N))
  PtrArray{T,N,R,S,NTuple{N,Nothing},NTuple{N,StaticInt{1}}}(p, sz, sx, o)
end
@inline function PtrArray(p::Ptr{T}, sz::S, sx::X, ::Val{R}) where {T,N,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},R}
  o = ntuple(Returns(static(1)), Val(N))
  PtrArray{T,N,R,S,X,NTuple{N,StaticInt{1}}}(p, sz, sx, o)
end
@inline function PtrArray0(p::Ptr{T}, sz::S, sx::X, ::Val{R}) where {T,N,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},R}
  o = ntuple(Returns(static(0)), Val(N))
  PtrArray{T,N,R,S,X,NTuple{N,StaticInt{1}}}(p, sz, sx, o)
end
@inline function PtrArray(p::Ptr{T}, sz::S) where {T,N,S<:Tuple{Vararg{Integer,N}}}
  sx = ntuple(Returns(nothing), Val(N))
  o = ntuple(Returns(static(1)), Val(N))
  R = ntuple(identity, Val(N))
  PtrArray{T,N,R,S,NTuple{N,Nothing},NTuple{N,StaticInt{1}}}(p, sz, sx, o)
end
@inline function PtrArray0(p::Ptr{T}, sz::S) where {T,N,S<:Tuple{Vararg{Integer,N}}}
  sx = ntuple(Returns(nothing), Val(N))
  o = ntuple(Returns(static(0)), Val(N))
  R = ntuple(identity, Val(N))
  PtrArray{T,N,R,S,NTuple{N,Nothing},NTuple{N,StaticInt{1}}}(p, sz, sx, o)
end
@inline function PtrArray(p::Ptr{T}, sz::S, sx::X) where {T,N,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}}}
  o = ntuple(Returns(static(1)), Val(N))
  R = ntuple(identity, Val(N))
  PtrArray{T,N,R,S,X,NTuple{N,StaticInt{1}}}(p, sz, sx, o)
end
@inline function PtrArray0(p::Ptr{T}, sz::S, sx::X) where {T,N,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}}}
  o = ntuple(Returns(static(0)), Val(N))
  R = ntuple(identity, Val(N))
  PtrArray{T,N,R,S,X,NTuple{N,StaticInt{1}}}(p, sz, sx, o)
end

@generated function _nondense_strides(
  strides::Tuple{Vararg{Integer}}, ::Val{F}, ::Val{D}
) where {F,D}
  t = Expr(:tuple)
  for i = eachindex(D)
    if D[i]
      push!(t.args, nothing)
    else
      x = Expr(:call,getfield,:strides,i)
      if F
        x = Expr(:call,>>>,x,static(3))
      end
      push!(t.args, x)
    end
  end
  Expr(:block,Expr(:meta,:inline),t)
end

@inline function LayoutPointers.stridedpointer(A::BitPtrArray)
  stridedpointer(getfield(A), StrideIndex(A))
end

@inline function PtrArray(
  ptr::StridedPointer{T,N,<:Any,0,R,<:Any,O},
  sz::S,
  ::Val{D},
) where {T,N,R,O,D,S}
  sx = _nondense_strides(strides(ptr), Val(true), sz)
  X = typeof(sx)
  PtrArray{T,N,R,S,X,O,T}(pointer(ptr), sz, sx, offsets(ptr))
end
@inline function PtrArray(
  ptr::StridedBitPointer{N,<:Any,0,R,<:Any,O},
  sz::S,
  ::Val{D},
) where {N,R,O,D,S}
  sx = _nondense_strides(strides(ptr), Val(false), sz)
  X = typeof(sx)
  PtrArray{Bool,N,R,S,X,O,Bit}(pointer(ptr), sz, sx, offsets(ptr))
end

const PtrVector{T,R,S,X,O} = PtrArray{T,1,R,S,X,O}
const PtrMatrix{T,R,S,X,O} = PtrArray{T,2,R,S,X,O}

@inline _sparse_strides(dd::Tuple{}, sx::Tuple{}) = ()
@inline function _sparse_strides(dd::Tuple{True,Vararg{Any,N}}, sx::Tuple{Integer,Vararg{Integer,N}}) where {N}
  (nothing, _sparse_strides(Base.tail(dd), Base.tail(sx))...)
end
@inline function _sparse_strides(dd::Tuple{False,Vararg{Any,N}}, sx::Tuple{Integer,Vararg{Integer,N}}) where {N}
  (first(sx), _sparse_strides(Base.tail(dd), Base.tail(sx))...)
end

@inline function PtrArray(
  p::Ptr{T}, sz::S, sx::X, offsets::O, ::Val{R}
) where {T,N,R,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Integer,N}}}
  AbstractPtrArray{T,N,R,S,X,O,T}(p, sz, sx, offsets)
end
@inline function PtrArray(
  p::Ptr{Bit}, sz::S, sx::X, offsets::O, ::Val{R}
) where {N,R,S<:Tuple{Vararg{Integer,N}},X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Integer,N}}}
  AbstractPtrArray{Bool,N,R,S,X,O,Bit}(p, sz, sx, offsets)
end
@inline function PtrArray(A::AbstractArray{T,N}) where {T,N}
  p = pointer(A)
  sz = size(A)
  sx = _sparse_strides(dense_dims(A), strides(A))
  R = map(Int, stride_rank(A))
  PtrArray(p, sz, sx, offsets(A), Val(R))
end

@inline Base.pointer(A::AbstractPtrStrideArray) = getfield(A, :ptr)

@inline Base.unsafe_convert(::Type{Ptr{T}}, A::AbstractStrideArray) where {T} =
  Base.unsafe_convert(Ptr{T}, pointer(A))
@inline Base.elsize(::AbstractStrideArray{T}) where {T} = sizeof(T)

@inline ArrayInterface.size(A::AbstractPtrStrideArray) = getfield(A, :sizes)
@inline function ArrayInterface.strides(A::AbstractPtrStrideArray{<:Any,<:Any,R}) where {R}
  _strides(size(A), getfield(A,:strides), Val{R}())
end
ArrayInterface.device(::AbstractStrideArray) = ArrayInterface.CPUPointer()

@generated function ArrayInterface.contiguous_axis(::Type{<:AbstractStrideArray{<:Any,<:Any,R,<:Any,X}}) where {R,X}
  i = findfirst(isone, R)
  C = i === nothing ? -1 : (X.parameters[i] === StaticInt{1} ? i : -1)
  StaticInt{C}()
end
ArrayInterface.contiguous_batch_size(
  ::Type{<:AbstractStrideArray}
) = StaticInt{0}()

ArrayInterface.known_size(::Type{<:AbstractStrideArray{<:Any,<:Any,<:Any,S}}) where {S} = Static.known(S)

@generated function ArrayInterface.stride_rank(
  ::Type{<:AbstractStrideArray{<:Any,<:Any,R}},
) where {R}
  t = Expr(:tuple)
  for r ∈ R
    push!(t.args, StaticInt{r}())
  end
  t
end
@generated function ArrayInterface.dense_dims(
  ::Type{<:AbstractStrideArray{<:Any,<:Any,<:Any,<:Any,X}},
) where {X}
  t = Expr(:tuple)
  for i = eachindex(X.parameters)
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
  ::Val{D},
  ) where {T,N,D}
  PtrArray0(p, s, _nondense_strides(x, Val(false), Val{D}()))
end
@inline function PtrArray(
  p::Ptr{T},
  s::Tuple{Vararg{Union{Integer,StaticInt},N}},
  x::Tuple{Vararg{Union{Integer,StaticInt},N}},
  ::Val{D},
  ) where {T,N,D}
  PtrArray(p, s, _nondense_strides(x, Val(false), Val{D}()))
end

@inline sparse_strides(A::AbstractPtrStrideArray) = getfield(A, :strides)

@inline function LayoutPointers.zero_offsets(A::AbstractPtrStrideArray{<:Any,N,R}) where {N,R}
  PtrArray(pointer(A), size(A), sparse_strides(A), ntuple(Returns(static(0)), Val(N)))
end

#=
PtrArray(ptr::Ptr, s::Tuple{Vararg{Union{Integer,StaticInt}}}, ::StaticInt{1}) =
  PtrArray(ptr, s)
PtrArray0(ptr::Ptr, s::Tuple{Vararg{Union{Integer,StaticInt}}}, ::StaticInt{1}) =
  PtrArray0(ptr, s)
@generated function contigperm(
  s::Tuple{Vararg{Union{Integer,StaticInt},N}},
  ::StaticInt{C},
) where {N,C}
  d = Expr(:tuple, Expr(:call, getfield, :s, C))
  perm = Expr(:tuple)
  resize!(perm.args, N)
  perm.args[C] = 1
  for n = 1:N
    if n != C
      push!(d.args, Expr(:call, getfield, :s, n))
      perm.args[n] = n + (n < C)
    end
  end
  Expr(:tuple, d, Expr(:call, Expr(:curly, :Val, perm)))
end
function PtrArray(
  ptr::Ptr,
  s::Tuple{Vararg{Union{Integer,StaticInt},N}},
  ::StaticInt{C},
) where {C,N}
  dim, perm = contigperm(s, static(C))
  permutedims(PtrArray(ptr, dim), perm)
end
function ptrarray0(
  ptr::Ptr,
  s::Tuple{Vararg{Union{Integer,StaticInt},N}},
  ::StaticInt{C},
) where {C,N}
  dim, perm = contigperm(s, static(C))
  permutedims(ptrarray0(ptr, dim), perm)
end
=#

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
@inline Base.stride(A::AbstractStrideArray, ::StaticInt{N}) where {N} = Base.stride(A, N::Int)
@generated _oneto(x) = Expr(:new, Base.OneTo{Int}, :(x % Int))

@inline create_axis(s, ::Zero) = CloseOpen(s)
@inline create_axis(s, ::One) = _oneto(unsigned(s))
@inline create_axis(::StaticInt{N}, ::One) where {N} = One():StaticInt{N}()
@inline create_axis(s, o) = CloseOpen(o, s + o)

@inline ArrayInterface.axes(A::AbstractStrideArray) = map(create_axis, size(A), offsets(A))
@inline Base.axes(A::AbstractStrideArray) = axes(A)
@inline Base.axes(A::AbstractStrideArray, d::StaticInt) = axes(A, d)


@generated function ArrayInterface.axes_types(
  ::Type{<:AbstractStrideArray{T,N,R,S,X,O}},
) where {S,T,N,R,X,O}
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


@inline ArrayInterface.offsets(A::PtrArray) = getfield(A, :offsets)
@inline ArrayInterface.static_length(A::AbstractStrideArray) = Static.reduce_tup(*, size(A))

# type stable, because index known at compile time
@inline type_stable_select(t::NTuple, ::StaticInt{N}) where {N} = getfield(t, N)
@inline type_stable_select(t::Tuple, ::StaticInt{N}) where {N} = getfield(t, N)
# type stable, because tuple is homogenous
@inline type_stable_select(t::NTuple, i::Integer) = getfield(t, i)
# make the tuple homogenous before indexing
@inline type_stable_select(t::Tuple, i::Integer) = getfield(map(Int, t), i)

@inline ArrayInterface._axes(A::AbstractStrideArray{S,D,T,N}, i::Integer) where {S,D,T,N} =
  __axes(A, i)
@inline ArrayInterface._axes(A::AbstractStrideArray{S,D,T,N}, i::Int) where {S,D,T,N} =
  __axes(A, i)
@inline ArrayInterface._axes(
  A::AbstractStrideArray{S,D,T,N},
  ::StaticInt{I},
) where {S,D,T,N,I} = __axes(A, StaticInt{I}())

@inline function __axes(
  A::AbstractStrideArray{T,N,R,S},
  i::Union{Integer,StaticInt},
) where {S,R,T,N}
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
@inline ArrayInterface.size(::AbstractStrideVector, ::StaticInt{N}) where {N} = One()
@inline ArrayInterface.size(A::AbstractStrideVector, ::StaticInt{1}) = length(A)
@inline ArrayInterface.size(A::AbstractStrideArray, ::StaticInt{N}) where {N} = size(A)[N]
@inline ArrayInterface.size(A::AbstractStrideArray, i::Int) = type_stable_select(size(A), i)
@inline Base.size(A::AbstractStrideArray, i::Union{Integer,StaticInt})::Int = size(A, i)

# Base.IndexStyle(::Type{<:AbstractStrideArray}) = IndexCartesian()
# Base.IndexStyle(::Type{<:AbstractStrideVector{<:Any,<:Any,<:Any,1}}) = IndexLinear()
@generated function Base.IndexStyle(
  ::Type{A},
) where {T,N,R,S,X,A<:AbstractStrideArray{T,N,R,S,X}}
  C = ArrayInterface.contiguous_axis(A)
  # if is column major || is a transposed contiguous vector
  if D===NTuple{N,Nothing} && (
    (isone(C) && R === ntuple(identity, Val(N))) ||
    (C === static(2) && R === (2, 1) && S <: Tuple{One,Integer})
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
  i::Tuple{Vararg{Union{Integer,StaticInt},NI}},
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
      :($ptr_expr + (first(i) - (ptr).si.offsets[1]) * $(static_sizeof(T))),
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
  i::Vararg{Union{Integer,StaticInt},N},
) where {S,D,N,C}
  fi = getfield(i, C) - getfield(offsets(A), C)
  u = pload(_offset_ptr(stridedpointer(A), i))
  (u >>> (fi & 7)) % Bool
end
@inline function Base.getindex(
  A::BitPtrArray{S,D,N,C},
  i::Union{Integer,StaticInt},
) where {S,D,N,C}
  j = i - oneunit(i)
  u = pload(reinterpret(Ptr{UInt8}, pointer(A)) + (j >>> 3))
  (u >>> (j & 7)) % Bool
end
@inline function Base.setindex!(
  A::BitPtrArray{S,D,N,C},
  v::Bool,
  i::Vararg{Union{Integer,StaticInt},N},
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
  i::Union{Integer,StaticInt},
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
Base.@propagate_inbounds Base.getindex(A::AbstractStrideVector, i::Integer, ::Colon) =
  view(A, i, :)
Base.@propagate_inbounds Base.getindex(A::AbstractStrideVector, ::Colon, ::Integer) = A

Base.@propagate_inbounds Base.getindex(A::AbstractStrideVector, i::Integer, ::Integer) =
  getindex(A, i)
Base.@propagate_inbounds function Base.getindex(
  A::AbstractStrideArray,
  i::Vararg{Union{Integer,StaticInt},K},
) where {K}
  b = preserve_buffer(A)
  GC.@preserve b begin
    PtrArray(A)[i...]
  end
end
Base.@propagate_inbounds function Base.getindex(
  A::AbstractStrideArray,
  i::Vararg{Union{Integer,StaticInt,Colon,AbstractRange},K},
) where {K}
  view(A, i...)
end
Base.@propagate_inbounds function Base.setindex!(
  A::AbstractStrideArray,
  v,
  i::Vararg{Union{Integer,StaticInt},K},
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
  A::PtrArray{Told,N},
) where {Tnew,Told,N}
  sz = let szt_old = static_sizeof(Told), szt_new = static_sizeof(Tnew)
    map(
      _scale,
      contiguous_axis_indicator(A),
      size(A),
      ntuple(Returns(szt_old), Val(N)),
      ntuple(Returns(szt_new), Val(N)),
    )
  end
  sp = reinterpret(Tnew, stridedpointer(A))
  PtrArray(sp, sz)
end

@generated function Base.reinterpret(
  ::typeof(reshape),
  ::Type{Tnew},
  A::PtrArray{Told,N,R,S,X,O},
) where {Told,Tnew,N,S,R,X,O}
  sz_old::Int = sizeof(Told)::Int
  sz_new::Int = sizeof(Tnew)::Int
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
      sz_n = Expr(:call, getfield, :s, n, false)
      sx_n = Expr(:call, getfield, :x, n, false)
      of_n = Expr(:call, getfield, :o, n, false)
      if (sz_old != sz_new) && (X.parameters[n] ≢ Nothing)
        sx_n = Expr(:call, ÷, Expr(:call, *, sx_n, static(sz_old)), static(sz_new))
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
    s = size(A)
    x = strides(A)
    o = offsets(A)
    PtrArray(p, $size_expr, $stride_expr, $offs_expr, Val{$Rnew}())
  end
end
@inline Base.reinterpret(::Type{T}, A::AbstractStrideArray) where {T} =
  StrideArray(reinterpret(T, PtrArray(A)), preserve_buffer(A))
@inline Base.reinterpret(::typeof(reshape), ::Type{T}, A::AbstractStrideArray) where {T} =
  StrideArray(reinterpret(reshape, T, PtrArray(A)), preserve_buffer(A))

Base.LinearIndices(x::AbstractStrideVector) = axes(x, static(1))
