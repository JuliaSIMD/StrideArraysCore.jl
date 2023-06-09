module StrideArraysCore
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
  @eval Base.Experimental.@max_methods 1
end

using LayoutPointers,
  StaticArrayInterface, ThreadingUtilities, ManualMemory, IfElse, Static
const ArrayInterface = StaticArrayInterface
using Static: StaticInt, StaticBool, True, False, Zero, One
const Integer = Union{StaticInt,Base.BitInteger}

using StaticArrayInterface:
  OptionallyStaticUnitRange,
  static_size,
  static_strides,
  offsets,
  indices,
  static_length,
  static_first,
  static_last,
  static_axes,
  dense_dims,
  stride_rank,
  StrideIndex,
  contiguous_axis_indicator
using LayoutPointers:
  AbstractStridedPointer,
  StridedPointer,
  StridedBitPointer,
  bytestrides,
  zstridedpointer,
  val_dense_dims,
  val_stride_rank,
  zero_offsets
using CloseOpenIntervals

using ManualMemory: preserve_buffer, offsetsize, MemoryBuffer

using SIMDTypes: NativeTypes, Bit

export PtrArray, StrideArray, StaticInt, static, @gc_preserve

@static if VERSION < v"1.7"
  struct Returns{T}
    x::T
  end
  (r::Returns)(args...) = r.x
end

const checkbounds_recompile = VERSION >= v"1.9.0" && Base.JLOptions().use_pkgimages == 1

@generated static_sizeof(::Type{T}) where {T} =
  :(StaticInt{$(Base.allocatedinline(T) ? sizeof(T) : sizeof(Int))}())
include("ptr_array.jl")
include("stridearray.jl")
include("thread_compatible.jl")
include("views.jl")
include("reshape.jl")
include("adjoints.jl")

if VERSION >= v"1.7.0" && hasfield(Method, :recursion_relation)
  dont_limit = Returns(true)
  for f in (_strides,)
    for m in methods(f)
      m.recursion_relation = dont_limit
    end
  end
end

if checkbounds_recompile
  function __init__()
    ccall(:jl_generating_output, Cint, ()) == 1 && return nothing
    if Base.JLOptions().check_bounds == 1
      @eval boundscheck() = true
    end
    #     # @require LoopVectorization="bdcacae8-1622-11e9-2a5c-532679323890" @eval using StrideArrays
  end
end

end
