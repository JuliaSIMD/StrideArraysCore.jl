mutable struct MemoryBuffer{L,T} <: DenseVector{T}
    data::NTuple{L,T}
    @inline function MemoryBuffer{L,T}(::UndefInitializer) where {L,T}
        @assert isbitstype(T) "Memory buffers must point to bits types, but `isbitstype($T) == false`."
        new{L,T}()
    end
end
@inline Base.unsafe_convert(::Type{Ptr{T}}, d::MemoryBuffer) where {T} = Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(d))
@inline MemoryBuffer{T}(::UndefInitializer, ::StaticInt{L}) where {L,T} = MemoryBuffer{L,T}(undef)
Base.size(::MemoryBuffer{L}) where L = (L,)
@inline Base.similar(::MemoryBuffer{L,T}) where {L,T} = MemoryBuffer{L,T}(undef)
# Base.IndexStyle(::Type{<:MemoryBuffer}) = Base.IndexLinear()
@inline function Base.getindex(m::MemoryBuffer{L,T}, i::Int) where {L,T}
    @boundscheck checkbounds(m, i)
    GC.@preserve m x = vload(pointer(m), VectorizationBase.lazymul(VectorizationBase.static_sizeof(T), i - one(i)))
    x
end
@inline function Base.setindex!(m::MemoryBuffer{L,T}, x, i::Int) where {L,T}
    @boundscheck checkbounds(m, i)
    GC.@preserve m vstore!(pointer(m), convert(T, x), lazymul(static_sizeof(T), i - one(i)))
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
@inline StrideArray(::UndefInitializer, s::Vararg{Integer,N}) where {N} = StrideArray{Float64}(undef, s)
@inline StrideArray(::UndefInitializer, ::Type{T}, s::Vararg{Integer,N}) where {T,N} = StrideArray{T}(undef, s)
@inline function StrideArray(A::PtrArray{S,D,T,N}, s::Tuple{Vararg{Integer,N}}) where {S,D,T,N}
  PtrArray(stridedpointer(A), s, val_dense_dims(A))
end
@inline function StrideArray(A::AbstractArray{T,N}, s::Tuple{Vararg{Integer,N}}) where {T,N}
  StrideArray(PtrArray(stridedpointer(A), s, val_dense_dims(A)), preserve_buffer(A))
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
    L = Base.allocatedinline(T) ? sizeof(T) : sizeof(Int)
    t = Expr(:tuple)
    for n ∈ 1:N
        push!(t.args, static_expr(L))
        L *= s.parameters[n].parameters[1]
    end
    Expr(:tuple, t, static_expr(L))
end
@generated function calc_strides_len(::Type{T}, s::Tuple{Vararg{Any,N}}) where {T, N}
    last_sx = :s_0
    st = Base.allocatedinline(T) ? sizeof(T) : sizeof(Int)
    q = Expr(:block, Expr(:meta,:inline), Expr(:(=), last_sx, static_expr(st)))
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
@inline VectorizationBase.preserve_buffer(A::StrideArray) = preserve_buffer(getfield(A, :data))

@inline PtrArray(A::StrideArray) = getfield(A, :ptr)

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
@inline zeroindex(A::PtrArray{S,D}) where {S,D} = PtrArray(zstridedpointer(A), size(A), Val{D}())
@inline zeroindex(A::StrideArray) = StrideArray(zeroindex(PtrArray(A)), preserve_buffer(A))

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
@inline function Base.similar(A::AbstractStrideArray, ::Type{T}) where {T}
    StrideArray{T}(undef, size(A))
end


@inline function Base.view(A::StrideArray, i::Vararg{Union{Integer,AbstractRange,Colon},K}) where {K}
    StrideArray(view(A.ptr, i...), A.data)
end
@inline function zview(A::StrideArray, i::Vararg{Union{Integer,AbstractRange,Colon},K}) where {K}
    StrideArray(zview(A.ptr, i...), A.data)
end
@inline function Base.permutedims(A::StrideArray, ::Val{P}) where {P}
    StrideArray(permutedims(A.ptr, Val{P}()), A.data)
end
@inline Base.adjoint(a::StrideVector) = StrideArray(adjoint(a.ptr), a.data)


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

