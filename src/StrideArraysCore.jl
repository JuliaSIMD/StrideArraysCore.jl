module StrideArraysCore

using VectorizationBase, ArrayInterface, ThreadingUtilities
using ArrayInterface: StaticInt, Zero, One, StaticBool, True, False,
    OptionallyStaticUnitRange, size, strides, offsets, indices,
    static_length, static_first, static_last, axes,
    dense_dims, stride_rank
using VectorizationBase: align, gep,
  AbstractStridedPointer, AbstractSIMDVector, vnoaliasstore!, staticm1,
    static_sizeof, StridedPointer, zstridedpointer,
    val_dense_dims, val_stride_rank, preserve_buffer

export PtrArray, StrideArray, StaticInt

include("closeopen.jl")
include("ptr_array.jl")
include("stridearray.jl")
include("thread_compatible.jl")
include("views.jl")
include("adjoints.jl")

function __init__()
    # @require LoopVectorization="bdcacae8-1622-11e9-2a5c-532679323890" @eval using StrideArrays
end

end
