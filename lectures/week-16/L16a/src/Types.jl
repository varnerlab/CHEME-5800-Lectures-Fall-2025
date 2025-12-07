abstract type AbstractNeuralNetwork end

# build a simple neural network model type -
struct MyFluxNeuralNetworkModel <: AbstractNeuralNetwork
    chain::Chain; # holds the model chain
end