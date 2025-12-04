
"""
    _energy(s::Array{<:Number,1}, W::Array{<:Number,2}, b::Array{<:Number,1}) -> Float32

Compute the Hopfield energy of state `s` given weight matrix `W` and bias vector `b` using
`E(s) = -0.5 * s' * W * s + b' * s`.
"""
function _energy(s::Array{<: Number,1}, W::Array{<:Number,2}, b::Array{<:Number,1})::Float32
    
    # initialize -
    tmp_energy_state = 0.0;
    number_of_states = length(s);

    # main loop -
    tmp = transpose(b)*s; # alias for the bias term
    for i ∈ 1:number_of_states
        for j ∈ 1:number_of_states
            tmp_energy_state += W[i,j]*s[i]*s[j];
        end
    end
    energy_state = -(1/2)*tmp_energy_state + tmp;

    # return -
    return energy_state;
end

"""
    ⊗(a::Array{T,1}, b::Array{T,1}) -> Array{T,2} where T <: Number

Compute the outer product of two numeric vectors `a` and `b`.

### Arguments
- `a::Array{T,1}`: a vector of length `m`.
- `b::Array{T,1}`: a vector of length `n`.

### Returns
- `Y::Array{T,2}`: matrix of size `m x n` with entries `Y[i,j] = a[i] * b[j]`.
"""
function ⊗(a::Array{T,1}, b::Array{T,1})::Array{T,2} where T <: Number

    # initialize -
    m = length(a)
    n = length(b)
    Y = zeros(m,n)

    # main loop 
    for i ∈ 1:m
        for j ∈ 1:n
            Y[i,j] = a[i]*b[j]
        end
    end

    # return 
    return Y
end


"""
    recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{Int32,1}, trueenergyvalue::Float32;
        maxiterations::Int = 1000, patience::Union{Int,Nothing} = nothing,
        miniterations_before_convergence::Union{Int,Nothing} = nothing) -> Tuple{Dict{Int64, Array{Int32,1}}, Dict{Int64, Float32}}

Run asynchronous Hopfield updates starting from `sₒ`, stopping on convergence, after `maxiterations`,
or once the energy drops below `trueenergyvalue`. Tracks the state and energy trajectory.

### Arguments
- `model::MyClassicalHopfieldNetworkModel`: Hopfield network parameters.
- `sₒ::Array{Int32,1}`: initial state (±1 spins).
- `trueenergyvalue::Float32`: early-stopping threshold; iteration halts when current energy is ≤ this value.
- `maxiterations::Int`: maximum updates before forcing termination.
- `patience::Union{Int,Nothing}`: buffer length used for equality-based convergence; defaults to `max(5, round(Int, 0.1 * N))` with `N` pixels.
- `miniterations_before_convergence::Union{Int,Nothing}`: minimum iterations before checking convergence; defaults to `patience` and is floored at `patience`.

### Returns
Tuple of dictionaries keyed from `0`:
- `frames::Dict{Int64, Array{Int32,1}}`: spin configuration per iteration.
- `energydictionary::Dict{Int64, Float32}`: energy per iteration.
"""
function recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{Int32,1}, trueenergyvalue::Float32;
    maxiterations::Int = 1000, patience::Union{Int,Nothing} = nothing,
    miniterations_before_convergence::Union{Int,Nothing} = nothing)::Tuple{Dict{Int64, Array{Int32,1}}, Dict{Int64, Float32}}

    # initialize -
    W = model.W; # get the weights
    b = model.b; # get the biases
    number_of_pixels = length(sₒ); # number of pixels
    patience_val = isnothing(patience) ? max(5, Int(round(0.1 * number_of_pixels))) : patience; # scale patience with problem size
    min_iterations = max(isnothing(miniterations_before_convergence) ? patience_val : miniterations_before_convergence, patience_val); # floor before declaring convergence
    S = CircularBuffer{Array{Int32,1}}(patience_val); # buffer to check for convergence
    
    # initialize -
    frames = Dict{Int64, Array{Int32,1}}(); # dictionary to hold frames
    energydictionary = Dict{Int64, Float32}(); # dictionary to hold energies
    has_converged = false; # convergence flag

    # setup -
    frames[0] = copy(sₒ); # copy the initial random state
    energydictionary[0] = _energy(sₒ,W, b); # initial energy
    s = copy(sₒ); # initial state
    iteration_counter = 1;
    while (has_converged == false)
        
        j = rand(1:number_of_pixels); # select a random pixel
        w = W[j,:]; # get the weights
        old_sj = s[j];
        h = dot(w,s) - b[j]; # state at node j
        
        # Edge case: if h == 0, we have a tie, so we randomly assign ±1
        if h == 0
            s[j] = rand() < 0.5 ? Int32(-1) : Int32(1); # random tie-break to avoid bias
        else
            s[j] = h > 0 ? Int32(1) : Int32(-1); # map sign to ±1 spins
        end

        energydictionary[iteration_counter] = _energy(s, W, b);
        state_snapshot = copy(s); # single snapshot reused for storage and convergence checks
        frames[iteration_counter] = state_snapshot;
        
        # check for convergence -
        push!(S, state_snapshot); # push the current state to the buffer
        if (length(S) == patience_val) && (iteration_counter >= min_iterations)
            all_equal = true;
            first_state = S[1]; # look at the oldest state in the buffer
            for state ∈ S
                if (hamming(first_state, state) != 0)
                    all_equal = false;
                    break;
                end
            end
            if (all_equal == true)
                has_converged = true; # we have converged
            end
        end
        
        # is energy below the true value?
        current_energy = energydictionary[iteration_counter];
        if (current_energy ≤ trueenergyvalue)
            has_converged = true; # stop
            @info "Energy value lower than true. Stopping"
        end

        # update counter, and check max iterations -
        iteration_counter += 1;
        if (iteration_counter > maxiterations && has_converged == false)
            has_converged = true; # we have reached the maximum number of iterations
            @warn "Maximum iterations reached without convergence."
        end

        
    end
            
    # return 
    frames, energydictionary
end



"""
    recover(model::MyModernHopfieldNetworkModel, sₒ::Array{T,1}; 
        maxiterations::Int64 = 1000, ϵ::Float64 = 1e-10) where T <: Number

Iteratively update a modern Hopfield network by alternating softmax probability updates and state reconstruction.
Stops when L1 change in probabilities is below `ϵ` or after `maxiterations`.

### Arguments
- `model::MyModernHopfieldNetworkModel`: Hopfield network containing memory matrix `X` and inverse-temperature `β`.
- `sₒ::Array{T,1}`: initial continuous state.
- `maxiterations::Int64`: maximum number of update steps (probability + state).
- `ϵ::Float64`: L1 threshold on successive probability vectors for convergence.

### Returns
- `s::Array{T,1}`: final state.
- `frames::Dict{Int64, Array{Float32,1}}`: stored states per iteration (keyed from `0`).
- `probability::Dict{Int64, Array{Float64,1}}`: stored probability vectors per iteration (keyed from `0`).
"""
function recover(model::MyModernHopfieldNetworkModel, sₒ::Array{T,1}; 
    maxiterations::Int64 = 1000, ϵₚ::Float64 = 1e-10, ϵₛ::Float64 = 1e-10) where T <: Number

    # initialize -
    X = model.X̂; # data matrix from the model. This holds the memories on the columns
    β = model.β; # beta parameter (inverse temperature)

    frames = Dict{Int64, Array{Float32,1}}(); # save the iterations -
    probability = Dict{Int64, Array{Float64,1}}(); # save the probabilities
    frames[0] = copy(sₒ); # copy the initial random state
    probability[0] = softmax(β*transpose(X)*sₒ); # initial probability
    should_stop_iteration = false; # flag to stop the iteration
    iteration_counter = 1; # iteration counter

    # loop -
    s = copy(sₒ); # initial state
    Δₚ = Inf; # initial delta for probability change
    Δₛ = Inf; # initial delta for state change
    while (should_stop_iteration == false)
        
        p = softmax(β*transpose(X)*s); # compute the probabilities
        s = X*p; # update the state
        
        frames[iteration_counter] = copy(s); # save a copy of the state in the frames dictionary
        probability[iteration_counter] = p; # save the probabilities in the probability dictionary

        # first: compute the difference between the current and previous probabilities
        if (iteration_counter > 1)
            Δₚ = (1/2)*norm(probability[iteration_counter] - probability[iteration_counter-1], 1); # L1 change
        end

        # next compute the difference between the current and previous states
        if (iteration_counter > 1)
            Δₛ = (1/2)*norm(frames[iteration_counter] - frames[iteration_counter-1], 2); # L2 change
        end

        # next: check for convergence. If we are out of iterations or the difference is small, we stop
        if (iteration_counter >= maxiterations)
            should_stop_iteration = true;
            @warn "Maximum iterations ($maxiterations) reached before convergence (Δ = $Δₚ, ϵ = $ϵₚ)."
        elseif (Δₚ ≤ ϵₚ && Δₛ ≤ ϵₛ)
            should_stop_iteration = true;
        else
            iteration_counter += 1; # increment the iteration counter, we are not done yet. Keep going.
        end
    end

    # return -
    return s,frames,probability
end


"""
    decode(simulationstate::Array{T,1}; number_of_rows::Int64 = 28, number_of_cols::Int64 = 28) -> Array{T,2}

Copy a flattened Hopfield state vector into a 2D array without remapping the values.
Entries are consumed in row-major order (first row left-to-right, then the next row), which differs from `reshape`.

- `simulationstate`: vector with at least `number_of_rows * number_of_cols` entries; values are preserved as-is.
- `number_of_rows`: output image height; defaults to 28 for MNIST-style digits.
- `number_of_cols`: output image width; defaults to 28 for MNIST-style digits.

Returns a `number_of_rows x number_of_cols` `Array{T,2}` filled with the copied values. Extra elements in longer vectors are ignored; a `BoundsError` is thrown if the vector is too short for the requested shape.
"""
function decode(simulationstate::Array{T,1}; 
    number_of_rows::Int64 = 28, number_of_cols::Int64 = 28)::Array{T,2} where T <: Number
    
    # initialize -
    reconstructed_image = Array{T,2}(undef, number_of_rows, number_of_cols);
    linearindex = 1;
    for row ∈ 1:number_of_rows
        for col ∈ 1:number_of_cols
            reconstructed_image[row,col] = simulationstate[linearindex];
            linearindex+=1;
        end
    end
    
    # return 
    return reconstructed_image
end
