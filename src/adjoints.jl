
@generated function Base.permutedims(
  A::AbstractPtrStrideArray{T,N,R,S,X,O},
  ::Val{P}
) where {P,T,N,R,S,X,O}
  length(P) == N || throw("cannot perm $N-dimensional array with $P of length = $(length(P))")
  q = Expr(
    :block,
    Expr(:meta,:inline),
    :(sz = $getfield(A,:sizes)),
    :(sx = $getfield(A, :strides)),
    :(o = $getfield(A,:offsets))
  )
  sz_expr = Expr(:tuple)
  sx_expr = Expr(:tuple)
  o_expr = Expr(:tuple)
  rv_expr = Expr(:tuple)
  for n = 1:N
    j = P[n]
    push!(sz_expr.args, :($getfield(sz ,$j)))
    push!(sx_expr.args, :($getfield(sx ,$j)))
    push!(o_expr.args, :($getfield(o ,$j)))
    push!(rv_expr.args, R[j])
  end
  push!(q.args, :(AbstractPtrArray(pointer(A), $sz_expr, $sx_expr, $o_expr, Val{$rv_expr}())))
  q
end


@inline Base.adjoint(A::AbstractStrideMatrix{<:Real}) = permutedims(A, Val{(2, 1)}())
@inline Base.transpose(A::AbstractStrideMatrix) = permutedims(A, Val{(2, 1)}())

@inline function Base.transpose(
  a::AbstractPtrArray{<:Any,1}
)
  AbstractPtrArray(pointer(a), (One(), static_length(a)), (StrideReset(One()), getfield(getfield(a,:strides), 1)), Val{(2,1)}())
end
@inline Base.adjoint(a::AbstractPtrArray{<:Real}) = transpose(a)


@inline row_major(a::AbstractStrideVector) = a
@inline row_major(A::AbstractStrideMatrix) = transpose(A)
@inline row_major(A::AbstractStrideArray{S,D,T,N}) where {S,D,T,N} =
  permutedims(A, Val(ntuple(Base.Fix1(-, N + 1), Val(N))))
