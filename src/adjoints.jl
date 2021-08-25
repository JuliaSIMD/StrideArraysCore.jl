
# For vectors
function permute_dims_expr(perm, D, C, B, R)
  s = Expr(:tuple) # size
  x = Expr(:tuple) # stride
  o = Expr(:tuple) # offsets
  Rnew = Expr(:tuple) # rank
  Dnew = Expr(:tuple) # dense
  Cnew = -1
  Bnew = -1
  N = length(perm)
  for n ∈ 1:N
    p = perm[n]
    push!(s.args, Expr(:ref, :s, p))
    push!(x.args, Expr(:ref, :x, p))
    push!(o.args, Expr(:ref, :o, p))
    push!(Rnew.args, R[p])
    push!(Dnew.args, D[p])
    if C == p
      Cnew = n
    end
    if B == p
      Bnew = n
    end
  end
  Dnew, Cnew, Bnew, Rnew, s, x, o
end
@generated function Base.permutedims(A::PtrArray{S,D,T,N,C,B,R}, ::Val{P}) where {S,D,T,N,C,B,R,P}
  Dnew, Cnew, Bnew, Rnew, s, x, o = permute_dims_expr(P, D, C, B, R)
  quote
    $(Expr(:meta,:inline))
    s = size(A)
    ptr = stridedpointer(A)
    x = strides(ptr)
    o = offsets(ptr)
    si = StrideIndex{$N,$Rnew,$Cnew}($x, $o)
    sp = stridedpointer(ptr.p, si, StaticInt{$B}())
    PtrArray(sp, $s, Val{$Dnew}())
  end
end

@inline Base.adjoint(A::AbstractStrideMatrix) = permutedims(A, Val{(2,1)}())
@inline Base.transpose(A::AbstractStrideMatrix) = permutedims(A, Val{(2,1)}())


@generated function Base.adjoint(a::PtrArray{S,D,T,1,C,B,R,X,O}) where {S,D,T,C,B,R,X,O}
  s = Expr(:tuple, :(One()), Expr(:ref, :s, 1))
  x₁ = Expr(:ref, :x, 1)
  x = Expr(:tuple, x₁, x₁)
  o = Expr(:tuple, :(One()), Expr(:ref, :o, 1))
  R1 = R[1]
  Rnew = Expr(:tuple, R1+1, R1)
  Dnew = Expr(:tuple, true, D[1])
  Cnew = C == 1 ? 2 : C
  quote
    $(Expr(:meta,:inline))
    s = size(a)
    ptr = stridedpointer(a)
    x = strides(ptr)
    o = offsets(ptr)
    si = StrideIndex{2,$Rnew,$Cnew}($x, $o)
    sp = stridedpointer(ptr.p, si, StaticInt{$B}())
    PtrArray(sp, $s, Val{$Dnew}())
  end
end

@inline Base.transpose(a::AbstractStrideVector) = adjoint(a)

@inline row_major(a::AbstractStrideVector) = a
@inline row_major(A::AbstractStrideMatrix) = A
@inline row_major(A::AbstractStrideArray{S,D,T,N}) where {S,D,T,N} = permutedims(A, Val(ntuple(Base.Fix1(-,N+1),Val(N))))

