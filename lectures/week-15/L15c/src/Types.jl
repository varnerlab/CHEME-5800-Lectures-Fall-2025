"""
Abstract supertype for Hopfield network models defined in this project.
"""
abstract type AbstractlHopfieldNetworkModel end

"""
    MyClassicalHopfieldNetworkModel <: AbstractlHopfieldNetworkModel

A mutable struct representing a classical Hopfield network model.

### Fields
- `W::Array{<:Number, 2}`: weight matrix.
- `b::Array{<:Number, 1}`: bias vector.
- `energy::Dict{Int64, Float32}`: energy of the states.
"""
mutable struct MyClassicalHopfieldNetworkModel <: AbstractlHopfieldNetworkModel

    # data -
    W::Array{<:Number, 2} # weight matrix
    b::Array{<:Number, 1} # bias vector
    energy::Dict{Int64, Float32} # energy of the states

    # empty constructor -
    MyClassicalHopfieldNetworkModel() = new();
end

"""
    MyModernHopfieldNetworkModel <: AbstractlHopfieldNetworkModel

A mutable struct representing a modern Hopfield network model.

### Fields
- `X::Array{<:Number, 2}`: data matrix with memories stored in the columns.
- `X̂::Array{<:Number, 2}`: normalized data matrix.
- `β::Number`: beta parameter (inverse temperature) controlling sharpness of softmax updates.
"""
mutable struct MyModernHopfieldNetworkModel <: AbstractlHopfieldNetworkModel

    # data -
    X::Array{<:Number, 2} # data matrix
    X̂::Array{<:Number, 2} # normalized data matrix
    β::Number; # beta parameter

    # empty constructor -
    MyModernHopfieldNetworkModel() = new();
end
