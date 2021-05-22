
struct CloseOpen{L <: Integer, U <: Integer} <: AbstractUnitRange{Int}
  start::L
  upper::U
  @inline CloseOpen{L,U}(l::L,u::U) where {L <: Integer, U <: Integer} = new{L,U}(l,u)
end
@inline CloseOpen(s::S, u::U) where {S,U} = CloseOpen{S,U}(s, u)
@inline CloseOpen(len::T) where {T<:Integer} = CloseOpen{Zero,T}(Zero(), len)

@inline Base.first(r::CloseOpen) = getfield(r,:start) % Int
@inline Base.first(r::CloseOpen{StaticInt{F}}) where {F} = F
@inline Base.step(::CloseOpen) = One()
# @inline Base.last(r::CloseOpen{<:Any,Int}) = getfield(r,:upper) - One()
@inline Base.last(r::CloseOpen{<:Integer,<:Integer}) = getfield(r,:upper)%Int - One()
@inline Base.last(r::CloseOpen{<:Integer,StaticInt{L}}) where {L} = L - 1
@inline ArrayInterface.static_first(r::CloseOpen) = getfield(r,:start)
@inline ArrayInterface.static_last(r::CloseOpen) = getfield(r,:upper) - One()
@inline Base.length(r::CloseOpen) = getfield(r,:upper) - getfield(r,:start)
@inline Base.length(r::CloseOpen{Zero}) = getfield(r,:upper)

@inline Base.iterate(r::CloseOpen) = (i = Int(first(r)); (i, i))
@inline Base.iterate(r::CloseOpen, i::Int) = (i += 1) ≥ r.upper ? nothing : (i, i)

ArrayInterface.known_first(::Type{<:CloseOpen{StaticInt{F}}}) where {F} = F
ArrayInterface.known_step(::Type{<:CloseOpen}) = 1
ArrayInterface.known_last(::Type{<:CloseOpen{<:Any,StaticInt{L}}}) where {L} = L - 1
ArrayInterface.known_length(::Type{CloseOpen{StaticInt{F},StaticInt{L}}}) where {F,L} = L - F

Base.IteratorSize(::Type{<:CloseOpen}) = Base.HasShape{1}()
Base.IteratorEltype(::Type{<:CloseOpen}) = Base.HasEltype()
@inline Base.size(r::CloseOpen) = (length(r),)
Base.eltype(::CloseOpen) = Int

