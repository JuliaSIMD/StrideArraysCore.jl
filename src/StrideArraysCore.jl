module StrideArraysCore

using LayoutPointers, ArrayInterface, ThreadingUtilities, ManualMemory
using ArrayInterface: StaticInt, Zero, One, StaticBool, True, False,
    OptionallyStaticUnitRange, size, strides, offsets, indices,
    static_length, static_first, static_last, axes,
    dense_dims, stride_rank, StrideIndex, offset1
using LayoutPointers:
  AbstractStridedPointer,
  StridedPointer, zstridedpointer,
  val_dense_dims, val_stride_rank

using ManualMemory: preserve_buffer

using SIMDTypes: NativeTypes

export PtrArray, StrideArray, StaticInt

@generated static_sizeof(::Type{T}) where {T} = :(StaticInt{$(Base.allocatedinline(T) ? sizeof(T) : sizeof(Int))}())
include("closeopen.jl")
include("ptr_array.jl")
include("stridearray.jl")
include("thread_compatible.jl")
include("views.jl")
include("adjoints.jl")

# function __init__()
#     # @require LoopVectorization="bdcacae8-1622-11e9-2a5c-532679323890" @eval using StrideArrays
# end

end
