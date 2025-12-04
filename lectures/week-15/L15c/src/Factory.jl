# -- PUBLIC METHODS BELOW HERE ---------------------------------------------------------------------------------------- #
"""
    build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple) -> MyClassicalHopfieldNetworkModel

Factory method for building a Hopfield network model. 

### Arguments
- `modeltype::Type{MyClassicalHopfieldNetworkModel}`: the type of the model to be built.
- `data::NamedTuple`: a named tuple containing the data for the model.

The named tuple should contain the following fields:
- `memories`: a matrix of memories (each column is a memory).

### Returns
- `model::MyClassicalHopfieldNetworkModel`: the built Hopfield network model with the following fields populated:
    - `W`: the weight matrix.
    - `b`: the bias vector.
    - `energy`: a dictionary of energies for each memory.
"""
function build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple)::MyClassicalHopfieldNetworkModel

    # initialize -
    model = modeltype();
    linearimagecollection = data.memories;
    number_of_rows, number_of_cols = size(linearimagecollection);
    W = zeros(Float32, number_of_rows, number_of_rows);
    b = zeros(Float32, number_of_rows); # zero bias for classical Hopfield

    # compute the W -
    for j ∈ 1:number_of_cols
        Y = ⊗(linearimagecollection[:,j], linearimagecollection[:,j]); # compute the outer product -
        W += Y; # update the W -
    end
    
    # no self-coupling and Hebbian scaling -
    for i ∈ 1:number_of_rows
        W[i,i] = 0.0f0; # no self-coupling in a classical Hopfield network
    end
    WN = (1/number_of_cols)*W; # Hebbian scaling by number of memories stored
    
    # compute the energy dictionary -
    energy = Dict{Int64, Float32}();
    for i ∈ 1:number_of_cols
        energy[i] = _energy(linearimagecollection[:,i], WN, b);
    end

    # add data to the model -
    model.W = WN;
    model.b = b;
    model.energy = energy;

    # return -
    return model;
end

"""
    build(modeltype::Type{MyModernHopfieldNetworkModel}, data::NamedTuple) -> MyModernHopfieldNetworkModel

Factory method for assembling a modern Hopfield network model from raw memories and an inverse-temperature parameter.

### Arguments
- `modeltype::Type{MyModernHopfieldNetworkModel}`: concrete model type to instantiate.
- `data::NamedTuple`: expects `memories` (matrix with memories on columns) and `β` (inverse-temperature scalar).

### Returns
- `model::MyModernHopfieldNetworkModel`: model populated with `X` (memory matrix) and `β`.
"""
function build(modeltype::Type{MyModernHopfieldNetworkModel}, data::NamedTuple)::MyModernHopfieldNetworkModel

    # initialize -
    model = modeltype();
    linearimagecollection = data.memories;
    normalized_linear_image_collection = data.normalized_memories;
    β = data.β; # beta parameter
    
    # add stuff the model -
    model.β = β;
    model.X = linearimagecollection;
    model.X̂ = normalized_linear_image_collection;

    # return -
    return model;
end
# --- PUBLIC METHODS ABOVE HERE --------------------------------------------------------------------------------------- #
