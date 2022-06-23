module StrideArraysCore

using LayoutPointers, ArrayInterface, ThreadingUtilities, ManualMemory, IfElse, Static
using Static: StaticInt, StaticBool, True, False, Zero, One
const Integer = Union{StaticInt,Base.BitInteger}

using ArrayInterface:
  OptionallyStaticUnitRange,
  size,
  strides,
  offsets,
  indices,
  static_length,
  static_first,
  static_last,
  axes,
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
  val_stride_rank
using CloseOpenIntervals

using ManualMemory: preserve_buffer, offsetsize, MemoryBuffer

using SIMDTypes: NativeTypes, Bit

export PtrArray, StrideArray, StaticInt, static

@generated static_sizeof(::Type{T}) where {T} =
  :(StaticInt{$(Base.allocatedinline(T) ? sizeof(T) : sizeof(Int))}())
include("ptr_array.jl")
include("stridearray.jl")
include("thread_compatible.jl")
include("views.jl")
include("reshape.jl")
include("adjoints.jl")

function __init__()
  ccall(:jl_generating_output, Cint, ()) == 1 && return nothing
  if Base.JLOptions().check_bounds == 1
    @eval boundscheck() = true
  end
  #     # @require LoopVectorization="bdcacae8-1622-11e9-2a5c-532679323890" @eval using StrideArrays
end

end
