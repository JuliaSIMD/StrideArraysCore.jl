mutable struct MemoryBuffer{L,T} <: DenseVector{T}
    data::NTuple{L,T}
    @inline function MemoryBuffer{L,T}(::UndefInitializer) where {L,T}
        @assert isbitstype(T) "Memory buffers must point to bits types, but `isbitstype($T) == false`."
        new{L,T}()
    end
end

@inline undef_memory_buffer(::Type{T}, ::StaticInt{L}) where {T,L} = MemoryBuffer{L,T}(undef)
@inline undef_memory_buffer(::Type{T}, L) where {T} = Vector{T}(undef, L)

struct StrideArray{S,D,T,N,C,B,R,X,O,A} <: AbstractStrideArray{S,D,T,N,C,B,R,X,O}
    ptr::PtrArray{S,D,T,N,C,B,R,X,O}
    data::A
end

@inline VectorizationBase.stridedpointer(A::StrideArray) = A.ptr.ptr

const StrideVector{S,D,T,C,B,R,X,O,A} = StrideArray{S,D,T,1,C,B,R,X,O,A}
const StrideMatrix{S,D,T,C,B,R,X,O,A} = StrideArray{S,D,T,2,C,B,R,X,O,A}

@inline StrideArray(A::AbstractArray) = StrideArray(PtrArray(A), A)

@inline function StrideArray{T}(::UndefInitializer, s::Tuple{Vararg{Integer,N}}) where {N,T}
    x, L = calc_strides_len(T,s)
    b = undef_memory_buffer(T, L ÷ static_sizeof(T))
    # For now, just trust Julia's alignment heuristics are doing the right thing
    # to save us from having to over-allocate
    # ptr = VectorizationBase.align(pointer(b))
    ptr = pointer(b)
    StrideArray(ptr, s, x, b, all_dense(Val{N}()))
end
@inline function StrideArray(ptr::Ptr{T}, s::S, x::X, b, ::Val{D}) where {S,X,T,D}
    StrideArray(PtrArray(ptr, s, x, Val{D}()), b)
end

function dense_quote(N::Int, b::Bool)
    d = Expr(:tuple)
    for n in 1:N
        push!(d.args, b)
    end
    Expr(:call, Expr(:curly, :Val, d))
end
@generated all_dense(::Val{N}) where {N} = dense_quote(N, true)

@generated function calc_strides_len(::Type{T}, s::Tuple{Vararg{StaticInt,N}}) where {T, N}
    L = sizeof(T)
    t = Expr(:tuple)
    for n ∈ 1:N
        push!(t.args, static_expr(L))
        L *= s.parameters[n].parameters[1]
    end
    Expr(:tuple, t, static_expr(L))
end
@generated function calc_strides_len(::Type{T}, s::Tuple{Vararg{Any,N}}) where {T, N}
    last_sx = :s_0
    q = Expr(:block, Expr(:meta,:inline), Expr(:(=), last_sx, static_expr(sizeof(T))))
    t = Expr(:tuple)
    for n ∈ 1:N
        push!(t.args, last_sx)
        new_sx = Symbol(:s_,n)
        push!(q.args, Expr(:(=), new_sx, Expr(:call, :vmul_fast, last_sx, Expr(:ref, :s, n))))
        last_sx = new_sx
    end
    push!(q.args, Expr(:tuple, t, last_sx))
    q
end

@inline VectorizationBase.preserve_buffer(A::MemoryBuffer) = A
@inline VectorizationBase.preserve_buffer(A::StrideArray) = preserve_buffer(A.data)

@inline maybe_ptr_array(A) = A
@inline maybe_ptr_array(A::AbstractArray) = maybe_ptr_array(ArrayInterface.device(A), A)
@inline maybe_ptr_array(::ArrayInterface.CPUPointer, A::AbstractArray) = PtrArray(A)
@inline maybe_ptr_array(_, A::AbstractArray) = A

@inline ArrayInterface.size(A::StrideArray) = getfield(getfield(A, :ptr), :size)

@inline VectorizationBase.bytestrides(A::StrideArray) = getfield(getfield(getfield(A, :ptr), :ptr), :strd)
@inline ArrayInterface.strides(A::StrideArray) = strides(getfield(A, :ptr))
@inline ArrayInterface.offsets(A::StrideArray) = getfield(getfield(getfield(A, :ptr), :ptr), :offsets)

@inline zeroindex(r::ArrayInterface.OptionallyStaticUnitRange{One}) = CloseOpen(Zero(), last(r))
@inline zeroindex(r::Base.OneTo) = CloseOpen(Zero(), last(r))
@inline zeroindex(r::AbstractUnitRange) = Zero():(last(r)-first(r))

@inline zeroindex(r::CloseOpen{Zero}) = r
@inline zeroindex(r::ArrayInterface.OptionallyStaticUnitRange{Zero}) = r


function gc_preserve_call(ex, skip=0)
    q = Expr(:block)
    call = Expr(:call, esc(ex.args[1]))
    gcp = Expr(:gc_preserve, call)
    for i ∈ 2:length(ex.args)
        arg = ex.args[i]
        if i+1 ≤ skip
            push!(call.args, arg)
            continue
        end
        A = gensym(:A); buffer = gensym(:buffer);
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

