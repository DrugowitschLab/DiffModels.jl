# Testing:
#
# - type hierarchy

using DiffModels
using Base.Test

@test ConstDrift <: AbstractDrift
@test VarDrift <: AbstractDrift

@test ConstBound <: AbstractBound
@test VarBound <: AbstractBound

@test SymBounds <: AbstractBounds
@test AsymBounds <: AbstractBounds
@test ConstSymBounds <: AbstractBounds
@test VarSymBounds <: AbstractBounds
@test ConstAsymBounds <: AbstractBounds
@test VarAsymBounds <: AbstractBounds