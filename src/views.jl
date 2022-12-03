"""
    rank_to_sortperm(::NTuple{N,Int}) -> NTuple{N,Int}
Returns the `sortperm` of the stride ranks.
"""
function rank_to_sortperm(R::NTuple{N,Int}) where {N}
  sp = ntuple(zero, Val{N}())
  r = ntuple(n -> sum(R[n] .≥ R), Val{N}())
  @inbounds for n = 1:N
    sp = Base.setindex(sp, n, r[n])
  end
  sp
end
rank_to_sortperm(R) = sortperm(R)

function view_quote(
  @nospecialize(I), N::Int, R::Vector{Int}, @nospecialize(X),
  st::Int, bit::Bool, zero_offsets::Bool
)
  q = Expr(:block, Expr(:meta,:inline), :(sz = $getfield(A,:sizes)), :(sx = $getfield(A, :strides)))
  zero_offsets || push!(q.args, :(o = $getfield(A,:offsets)))
  # stride rank of (2, 3, 1) means we iter j = (3, 1, 2)
  p_expr = :($getfield(A,:ptr))
  sz_expr = Expr(:tuple)
  so_expr = Expr(:tuple)
  r_perm = Int[]#Expr(:tuple)
  i_off = prev_stride = :null
  n = N
  i_off_n_sym::Union{Nothing,Symbol} = nothing
  X_p0 = nothing
  while true
    j = findfirst(==(n), R)::Int
    I_n = I[j]
    s_n = Symbol(:s_,n)
    push!(q.args, Expr(:(=), s_n, Expr(:call, getfield, :sz, j)))
    x_n = Symbol(:x_,n)
    X_j = X.parameters[j]
    x_nothing = X_j === Nothing
    x_nothing || push!(q.args, Expr(:(=), x_n, Expr(:call, getfield, :sx, j)))
    o_n = Symbol(:o_,n)
    if zero_offsets
      push!(q.args, Expr(:(=), o_n, static(0)))
    else
      push!(q.args, Expr(:(=), o_n, Expr(:call, getfield, :o, j)))
    end
    # scale i_off if not null
    if i_off ≢ :null
      if X_p0 === Nothing
        # scale i_off by current-dim size
        i_off_scaled = Symbol(i_off, :_scaled)
        push!(q.args, Expr(:(=), i_off_scaled, Expr(:call, *, i_off, s_n)))
        i_off = i_off_scaled
      else
        # scale by i_off with previous-dim stride
        prev_stride === :null && throw("prev_stride ≡ :null, but that shouldn't be possible when i_off ≢ :null")
        i_off_expr = Expr(:call, *, i_off, prev_stride)
        i_off_expr = bit ? Expr(:call, >>>, i_off_expr, 3) : Expr(:call, *, i_off_expr, st)
        p_expr = Expr(:call, +, p_expr, i_off_expr)
        i_off = prev_stride = :null
      end
    end
    # index with current dim
    if I_n <: Integer
      i_n = Symbol(:i_,n)
      push!(q.args, Expr(:(=), i_n, Expr(:call, getfield, :i, j)))
      i_off_n_sym = i_n
    else
      pushfirst!(r_perm, j)
      pushfirst!(so_expr.args, o_n)
      if I_n === Colon
        # we keep this dimension
        pushfirst!(sz_expr.args, s_n)
      else#if I_n <: AbstractRange
        i_n = Symbol(:i_,n)
        push!(q.args, Expr(:(=), i_n, Expr(:call, getfield, :i, j)))
        i_first_n = Symbol(:i_first_,n)
        push!(q.args, Expr(:(=), i_first_n, Expr(:call, static_first, i_n)))
        i_off_n_sym = i_first_n
        dim_n = Symbol(:dim_,n)
        fast_len = if (I_n <: AbstractUnitRange) || ArrayInterface.known_step(I_n) === 1
          fast_len = Expr(
            :call, +,
            Expr(:call, -, Expr(:call, static_last, i_n), i_first_n), static(1)
          )
        else
          step_n = Symbol(:step_,n)
          push!(q.args, Expr(:(=), step_n, Expr(:call, ArrayInterface.static_step, i_n)))
          Expr(
            :call, +, static(1),
            Expr(:call, ÷, Expr(:call, -, Expr(:call, static_last, i_n), i_first_n), step_n)
          )
        end
        push!(q.args, Expr(:(=), dim_n, fast_len))
        pushfirst!(sz_expr.args, dim_n)
      end      
    end
    if i_off_n_sym ≢ nothing
      i_off_n = Symbol(:ioff_,n)
      if zero_offsets
        if i_off === :null
          push!(q.args, Expr(:(=), i_off_n, i_off_n_sym))
        else
          push!(q.args, Expr(:(=), i_off_n, Expr(:call, +, i_off, i_off_n_sym)))
        end
      else
        i_off_expr = Expr(:call, -, i_off_n_sym, o_n)
        if i_off === :null
          push!(q.args, Expr(:(=), i_off_n, i_off_expr))
        else
          push!(q.args, Expr(:(=), i_off_n, Expr(:call, +, i_off, i_off_expr)))
        end
      end
      i_off = i_off_n
      i_off_n_sym = nothing
    end
    n == 1 && break
    prev_stride = x_n
    n -= 1
    X_p0 = X_j
  end
  # build strides going forward
  sx_expr = Expr(:tuple)
  if length(sz_expr.args) > 0
    # if the output array is dim-0, we can skip all of this
    j_prev = findfirst(==(1), R)::Int
    I_p = I[j_prev]
    X_p = X.parameters[j_prev]
    I_p_colon = I_p === Colon
    prev_nonunit_step::Bool = (I_p <: AbstractRange) && (!(I_p <: AbstractUnitRange)) && (ArrayInterface.known_step(I_p) ≢ 1)
    prev_stride = :null # only used for dropped/integer dims
    if prev_nonunit_step
      stride_expr = Expr(:call, ArrayInterface.static_step, :i_1)
      if X_p ≢ Nothing
        stride_expr = Expr(:call, *, stride_expr, :x_1)
      end
      push!(q.args, Expr(:(=), prev_stride, stride_expr))
      push!(sx_expr, stride_expr)
      prev_nonunit_step = true
    elseif !(I_p <: Integer)
      if X_p ≢ Nothing
        push!(sx_expr.args, :x_1)
      else
        push!(sx_expr.args, nothing)
      end
    else
      prev_stride = :first_stride
      if X_p === Nothing
        push!(q.args, Expr(:(=), prev_stride, static(1)))
      else
        push!(q.args, Expr(:(=), prev_stride, :x_1))
      end
    end
    for n = 2:N
      j = findfirst(==(n), R)::Int
      x_n = Symbol(:x_, n)
      I_n = I[j]
      X_n = X.parameters[j]
      x_nothing = X_n === Nothing
      I_n_colon = I_n === Colon
      if I_n <: Integer
        if !prev_nonunit_step
          prev_stride_n = Symbol(:prev_stride_, n)
          if prev_stride === :null
            if x_nothing
              push!(q.args, Expr(:(=), prev_stride_n, Symbol(:s_,n-1)))
            else
              push!(q.args, Expr(:(=), prev_stride_n, x_n))
            end
          else
            if x_nothing
              push!(q.args, Expr(:(=), prev_stride_n, Expr(:call, *, prev_stride, Symbol(:s_,n-1))))
            else
              push!(q.args, Expr(:(=), prev_stride_n, Expr(:call, *, prev_stride, x_n)))
            end
          end
          prev_stride = prev_stride_n
        end
      else
        nonunit_step = !((I_n_colon) || (I_n<:AbstractUnitRange) || (ArrayInterface.known_step(I_n) === 1))
        if prev_nonunit_step
          sr_expr = Expr(:call, getfield, Expr(:call, strides, :A), j)
          prev_stride = Symbol(:prev_stride_, n)
          push!(q.args, Expr(:(=), prev_stride, sr_expr))
          push!(sx_expr.args, Expr(:call, StrideReset, prev_stride))
        else
          if prev_stride !== :null
            if x_nothing
              x_nothing = false
              push!(q.args, Expr(:(=), x_n, Expr(:call, *, Symbol(:s_,n-1), prev_stride)))
            else
              x_n_scaled = Symbol(x_n, :_scaled)
              push!(q.args, Expr(:(=), x_n_scaled, Expr(:call, *, x_n, prev_stride)))
              x_n = x_n_scaled
            end
          end
          if nonunit_step
            step_n = Symbol(:step_,n)
            if x_nothing
              push!(sx_expr.args, Expr(:call, :*, Symbol(:s_, n-1), step_n))
            else
              push!(sx_expr.args, Expr(:call, :*, x_n, step_n))
            end
          else
            if !x_nothing
              push!(sx_expr.args, x_n)
            elseif I_p_colon
              push!(sx_expr.args, nothing)
            else
              push!(sx_expr.args, Symbol(:s_, n-1))
            end
          end
        end
        prev_stride = :null
        prev_nonunit_step = nonunit_step
      end
      j_prev = j
      I_p = I_n
      I_p_colon = I_n_colon
    end
  end
  if i_off ≢ :null
    i_off_expr = bit ? Expr(:call, >>>, i_off, 3) : Expr(:call, *, i_off, st)
    p_expr = Expr(:call, +, p_expr, i_off_expr)
  end
  array_expr = :(AbstractPtrArray($p_expr, $sz_expr, $sx_expr, $so_expr))
  # make counters dense, i.e. [2, 5, 3] -> [1, 3, 2]
  r_perm = map(y->sum(<=(y), r_perm), r_perm)
  if r_perm != eachindex(r_perm)
    r_expr = Expr(:tuple)
    for r = r_perm
      push!(r_expr.args, r)
    end
    array_expr = :($permutedims($array_expr, Val{$r_expr}()))
  end
  push!(q.args, array_expr)
  return q
end

# @inline _isascending(::Tuple{}, _) = true
# @inline function _isascending(x::Tuple, prev)
#   next = first(x)
#   next > prev && _isascending(Base.tail(x), next)
# end

@generated function Base.view(
  A::AbstractPtrArray{T,N,R,S,X,O,P},
  i::Vararg{Union{Integer,AbstractRange,Colon},N},
) where {T,N,R,S,X,O,P}
  r = Vector{Int}(undef, N)
  for n = 1:N
    r[n] = R[n]
  end
  view_quote(i, N, r, X, sizeof(T), P===Bit, false)
end
@generated function zview(
  A::AbstractPtrArray{T,N,R,S,X,O,P},
  i::Vararg{Union{Integer,AbstractRange,Colon},N},
) where {T,N,R,S,X,O,P}
  r = Vector{Int}(undef, N)
  for n = 1:N
    r[n] = R[n]
  end
  view_quote(i, N, r, X, sizeof(T), P===Bit, true)
end

@inline Base.view(A::AbstractPtrArray, ::Colon) = vec(A)
@inline zview(A::AbstractPtrArray, ::Colon) = vec(A)

@inline Base.view(A::AbstractPtrVector, ::Colon) = A
@inline zview(A::AbstractPtrVector, ::Colon) = A

@inline Base.SubArray(
  A::AbstractStrideArray,
  i::Tuple{Vararg{Union{Integer,AbstractRange,Colon},K}},
) where {K} = view(A, i...)
