# Testing:
#
# - type hierarchy

using DiffModels
using Test

@test ConstDrift <: AbstractDrift
@test VarDrift <: AbstractDrift

@test ConstBound <: AbstractBound
@test VarBound <: AbstractBound
@test LinearBound <: AbstractBound

@test SymBounds <: AbstractBounds
@test AsymBounds <: AbstractBounds
@test ConstSymBounds <: AbstractBounds
@test ConstSymBounds <: ConstBounds
@test VarSymBounds <: AbstractBounds
@test ConstAsymBounds <: AbstractBounds
@test ConstAsymBounds <: ConstBounds
@test VarAsymBounds <: AbstractBounds
