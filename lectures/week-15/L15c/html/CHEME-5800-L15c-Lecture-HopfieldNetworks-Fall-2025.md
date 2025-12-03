# L15c: Modern Hopfield Networks and the Single-Head Attention Mechanism
In the previous lecture, we introduced classical Hopfield networks as associative memory systems that encode binary patterns using Hebbian learning. While elegant, classical networks have significant limitations in storage capacity and pattern representation.

In this lecture, we extend our understanding to modern Hopfield networks, which overcome these limitations and provide a bridge to contemporary deep learning architectures.

> __Learning Objectives:__
> 
> By the end of this lecture, you should be able to define and explain:
>
> * __Modern Hopfield Energy Function__: Explain how the log-sum-exp energy function generalizes the classical Hopfield energy to enable storage of continuous (not just binary) patterns. Understand the role of the inverse temperature parameter $\beta$ in controlling convergence sharpness and the connection between log-sum-exp and softmax operations.
> * __Exponential Storage and Convergence__: Describe how modern Hopfield networks achieve exponentially larger storage capacity compared to the classical $0.138N$ limit. Explain the memory retrieval algorithm using softmax-weighted updates and understand why modern networks exhibit exponential convergence (typically 1–5 iterations) rather than the polynomial convergence of classical networks.
> * __Connection to Attention Mechanisms__: Recognize that the modern Hopfield update rule is mathematically equivalent to single-head attention in transformer architectures. Understand how queries, keys, and values in attention correspond to states and memories in Hopfield networks, bridging associative memory theory with modern deep learning.

Let's get started!
___

## Examples
Today, we will use the following examples to illustrate key concepts:

> [▶ Analyze a modern Hopfield network](CHEME-5800-L15c-Example-ModernHopfieldNetwork-Fall-2025.ipynb). In this example, we analyze a modern Hopfield Network to understand how it encodes and retrieves gray-scale (continuous) patterns. This example builds on the concepts introduced in the previous lecture and demonstrates the application of modern Hopfield networks in continuous associative memory tasks.
___

## Origin story: McCulloch-Pitts Neurons
In [their paper, McCulloch and Pitts (1943)](https://link.springer.com/article/10.1007/BF02478259) explored how the brain could produce highly complex patterns by using many [interconnected _basic cells (neurons)_](https://en.wikipedia.org/wiki/Biological_neuron_model). McCulloch and Pitts suggested a _highly simplified model_ of a neuron. Nevertheless, they made a foundational contribution to developing artificial neural networks that we find in wide use today. Let's look at the model of a neuron proposed by McCulloch and Pitts.

Suppose we have a neuron that takes an input vector $\mathbf{n}(t) = (n^{(t)}_1, n^{(t)}_2, \ldots, n^{(t)}_{m})$, where each component $n_k\in\mathbf{n}$ is a binary value (`0` or `1`) representing the state of other predecessor neurons $n_1,n_2,\ldots,n_m$ at time $t$. Then, the state of our neuron (say neuron $k$) at time $t+1$ is given by:
$$
\begin{align*}
n_{k}(t+1) &= \sigma\left(\sum_{j=1}^{m} w_{kj} n_j(t) - \theta_k\right) \\
\end{align*}
$$
where $\sigma:\mathbb{R}^{n}\rightarrow\mathbb{R}$ is an _activation function_ that maps the weighted sum of a vector of inputs to a scalar (binary) output. In the original paper, the state of neuron $k$ at time $t+1$, denoted as $n_k(t+1)\in\{0,1\}$, where $w_{kj}$ is the weight of the connection from predecessor neuron $j$ to neuron $k$, and $\theta_k$ is the threshold for neuron $k$. 
* __Activation function__: In the original McCulloch and Pitts model, the activation function $\sigma$ is a step function, which means that the output of the neuron is `1` if the weighted sum of inputs exceeds the threshold $\theta_k$, and `0` otherwise. In other words, the neuron "fires" (produces an output of `1`) if the total input to the neuron is greater than or equal to the threshold $\theta_k$. This is a binary output, simplifying real biological neurons that can produce continuous outputs.
* __Parameters__: The weights $w_{kj}\in\mathbb{R}$ and the threshold $\theta_k\in\mathbb{R}$ are parameters of the neuron that determine its behavior. The weights can be positive or negative, representing the strength and direction of the influence of the input neurons on the output neuron. The threshold determines how much input the neuron needs to "fire" (i.e., produce an output of `1`).

While the McCulloch-Pitts neuron model simplifies real biological neurons, it laid the groundwork for the development of more complex artificial neural networks. The key idea is that by combining many simple neurons in a network, we can create complex functions and learn to approximate any continuous function. This idea is at the heart of modern deep learning and neural networks. 

__Hmmmm__. These ideas _really_ seem familiar. Have we seen this before? Yes! The McCulloch-Pitts Neuron underpins [The Perceptron (Rosenblatt, 1957)](https://en.wikipedia.org/wiki/Perceptron), [Hopfield networks](https://en.wikipedia.org/wiki/Hopfield_network), and [Boltzmann machines](https://en.wikipedia.org/wiki/Boltzmann_machine). Wow!!

___

## Classical Hopfield Networks
A classical Hopfield network is a fully connected, undirected graph consisting of $N$ nodes (neurons), where each node has a binary state $s = \pm 1$. Each node is connected to every other node, but not to itself. The connection weights between nodes $i$ and $j$, denoted $w_{ij} \in \mathbf{W}$, are determined using a **Hebbian learning rule**.

> __Hebbian learning__
>
> * __Hebbian Learning Rule__: The [Hebbian learning rule](https://en.wikipedia.org/wiki/Hebbian_theory), proposed by [Donald Hebb in 1949](https://en.wikipedia.org/wiki/Donald_O._Hebb), states that synaptic connections between neurons are strengthened when they activate (fire) simultaneously, forming the biological basis for __associative learning__. This "fire together, wire together" principle underpins unsupervised learning in neural networks, linking co-active nodes to enable pattern storage and recall.
> * __Different?__ Unlike the previous examples of learning, e.g., logistic regression or any of the online learning approaches that we looked at previously, the parameters (weights) in a [Hopfield network](https://en.wikipedia.org/wiki/Hopfield_network) are entirely specified by the memories we want to encode. Thus, we do not need to search for weights or learn them by experimenting with the world. Instead, we can directly compute the weights from the memories we want to encode.
> * __Recurrent?__ A Hopfield network is a special type of [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) in which recurrence is used to settle into a stable pattern iteratively. It is considered recurrent because its units are symmetrically and recurrently connected, allowing the network to evolve toward an energy minimum over time.
> 
> The Hebbian learning rule uses only local computations without explicit training iterations, providing a biological basis for memory encoding that operates without specialized hardware or extended optimization cycles.


### Encoding memories into a Hopfield network
Suppose we wish our network to memorize $K$ images, where each image is an $n\times{n}$ collection of black and white pixels represented as a vector $\mathbf{s}_{i}\in\left\{-1,1\right\}^{n^2}$. We encode the image using the following rule: if the pixel is white, we set the memory value to `1`, and if the pixel is black, we set the memory value to `-1`. Then, the weights that encode these $K$ images are given by:
$$
\begin{equation*}
\mathbf{W} = \frac{1}{K}\cdot\sum_{i=1}^{K}\mathbf{s}_{i}\otimes\mathbf{s}_{i}^{\top}
\end{equation*}
$$
where $\mathbf{s}_{i}$ denotes the state (pixels) of the image we want to memorize, and $\otimes$ denotes the outer product. Thus, the weights are like an average of all of our memories!

> __How big can $K$ be?__: The maximum theoretical storage limit $K_{\text{max}}$ of a classical Hopfield network, using the standard Hebbian learning rule, is approximately $K_{max}\sim{0.138}{N}$, where $N$ is the number of neurons in the network. Thus, the network can reliably store about 14% of its size in patterns before retrieval errors become significant due to interference between stored patterns.

Suppose we've encoded $K$ images and want to retrieve one of them. This seems magical. How does it work? 

### Algorithm: Memory retrieval
Each memory in a Hopfield network is encoded as a _local minimum_ of a global energy function. Thus, during memory retrieval, when we supply a random state vector $\hat{\mathbf{s}}$, we will recover the _closest_ memory encoded in the network to where we start.
The overall energy of the network is given by:
$$
\begin{equation*}
E(\mathbf{s}) = -\frac{1}{2}\,\sum_{ij}w_{ij}s_{i}s_{j} - \sum_{i}b_{i}s_{i}
\end{equation*}
$$
where $w_{ij}\in\mathbf{W}$ are the weights of the network, and $b_{i}$ is a bias term (typically set to zero but can be used to control the activation threshold of the neurons).

Let's outline some pseudocode for the memory retrieval algorithm.

__Initialize__: Compute the weights $w_{ij}\in\mathbf{W}$ using the Hebbian learning rule. Initialize the network with a random state $\mathbf{s}$. Set $\texttt{converged}\gets\texttt{false}$, the iteration counter $t\gets{1}$, maximum iterations $\texttt{maxiter} = 10N$ (where $N$ is the number of neurons), and patience parameter $\texttt{patience}$.

> **Patience Parameter** 
> 
> The patience parameter determines how many consecutive identical states are required to declare convergence. It is a practical heuristic that balances convergence detection with computational efficiency. Classical Hopfield networks can occasionally get stuck in short oscillation cycles (e.g., alternating between a few states). Requiring a fixed number of consecutive identical states ensures the network has truly converged to a stable attractor rather than just pausing briefly or terminating prematurely. 

__Track__: Initialize a queue $\texttt{S}$ to store the last $\texttt{patience}$ state vectors.

While not $\texttt{converged}$ __do__:
1. Store the current state: $\mathbf{s}_{\text{old}} \gets \mathbf{s}$.
2. **Asynchronous update**: Choose a random node $i$ and compute a new state $s_{i}^{\prime}$ using the update rule: $s_{i}^{\prime} \leftarrow \texttt{sign}\left(\sum_{j}w_{ij}s_{j}-b_{i}\right)$, where $\texttt{sign}(\cdot)$ is the sign function and $b_{i}$ is a bias (threshold) parameter.
3. Update the network state: $\mathbf{s} \leftarrow \mathbf{s}^{\prime}$ (only neuron $i$ changes).
4. Add current state to history: $\texttt{S}\gets\texttt{S} \cup \{\mathbf{s}\}$.
5. **Check for convergence**: There are several criteria we can use to stop the iteration and determine if the network has converged:
   - **State stability**: If the state history $\texttt{S}$ contains $\texttt{patience}$ states and all states in the history are identical (Hamming distance = 0 between all consecutive pairs), then set $\texttt{converged}\gets\texttt{true}$.
   - **Memory retrieval**: Alternatively, if the current state $\mathbf{s}$ exactly matches any stored memory pattern $\mathbf{s}_k$ (Hamming distance = 0), then set $\texttt{converged}\gets\texttt{true}$.
   - **Energy minimum reached**: If the energy $E(\mathbf{s})$ equals or falls below the __true minimum__, then set $\texttt{converged}\gets\texttt{true}$.
   - __Max iterations__: If $t \geq \texttt{maxiter}$, set $\texttt{converged}\gets\texttt{true}$. Notify that maximum iterations reached without convergence.
6. If the length of the state history queue $\texttt{S}$ exceeds $\texttt{patience}$ length, remove the oldest state.
7. Update iteration counter: $t \leftarrow t + 1$.

> **Hamming Distance**: The Hamming distance between two binary vectors $\mathbf{a}$ and $\mathbf{b}$ is defined as $H(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{N} \mathbb{I}[a_i \neq b_i]$, where $\mathbb{I}[\cdot]$ is the indicator function. For convergence, we check if $H(\mathbf{s}_{\text{current}}, \mathbf{s}_{\text{previous}}) = 0$, meaning the states are identical.

### Convergence
Classical Hopfield networks have strong theoretical convergence guarantees that make them particularly appealing for associative memory tasks.

* **Guaranteed Convergence**: The asynchronous update rule ensures that the network's energy function $E(\mathbf{s})$ is monotonically non-increasing with each neuron update. Since the state space is finite (each neuron can only be in one of two states: $\pm 1$), and the energy has a lower bound, the network is **guaranteed to converge** to a stable state or a short limit cycle.
* **Energy Landscape**: Each stored memory pattern $\mathbf{s}_k$ corresponds to a local minimum in the energy landscape. When the network is initialized with a partial or noisy version of a stored pattern, the iterative updates guide the system downhill in energy space toward the nearest local minimum, effectively "cleaning up" the corrupted input and retrieving the complete memory.

The classical Hopfield network's convergence properties make it a robust model for associative memory, capable of retrieving stored patterns from incomplete or noisy inputs, provided the number of stored patterns does not exceed the network's capacity.

> **Convergence** 
>
> In practice, classical Hopfield networks typically converge quickly: in the best case, when initialized close to a stored pattern, convergence can occur in $\mathcal{O}(1)$ steps. However, convergence time varies based on network size and the number of stored patterns. In the worst case, convergence takes $\mathcal{O}(N^2)$ steps, though this is rare in practice.
>
> **Limitations**: While convergence is guaranteed, the network may converge to:
> - **Spurious attractors**: Stable states that are not stored memories but arise from interference between patterns
> - **Incomplete patterns**: Local minima that represent corrupted versions of stored memories
> - **Wrong memories**: The network may converge to a different stored pattern than intended if the initial state is equidistant from multiple memories
> - **Antipatterns**: The network may converge to the bitwise inverse of a stored memory (e.g., if $\mathbf{s}_k$ is stored, then $-\mathbf{s}_k$ is also a stable attractor). This occurs because the Hebbian learning rule creates symmetric energy wells around both a pattern and its inverse.

The convergence behavior degrades as the number of stored patterns approaches the theoretical limit of $K \approx 0.138N$, where pattern interference becomes significant and spurious attractors multiply.

Let's look at an example to illustrate these concepts.

> __Example:__
> 
> [▶ Analyze a classical Hopfield Network](CHEME-5800-L15a-Example-HopfieldNetworks-Fall-2025.ipynb). In this example, we analyze an example of a classical Hopfield Network to understand how it encodes and retrieves binary patterns using Hebbian learning and asynchronous updates. We consider uncorrelated binary patterns and investigate the network's ability to recover original patterns from noisy inputs.
____

## Modern Hopfield Networks
Classical Hopfield networks are elegant but limited: they can only store a small number of binary patterns reliably. Modern Hopfield networks address these shortcomings.

> __Why are these interesting?__ The modern Hopfield energy function generalizes that of the classical network to allow the storage of *continuous* (not just binary) patterns. It also enables the storage of exponentially many (potentially *correlated*) memories and exhibits much faster convergence behavior than classical networks.

The key innovation in modern Hopfield networks is the reformulation of the energy function. [Krotov and Hopfield (2016)](https://arxiv.org/abs/1606.01164) proposed a new energy function of the form:
$$
\begin{align*}
E(\mathbf{s}) &= -\sum_{i=1}^{K}F(\underbrace{\mathbf{m}_{i}^{\top}\mathbf{s}}_{\text{similarity}}) \\
\end{align*}
$$
where $F$ is a nonlinear function, $\mathbf{m}_i$ is the $i$-th memory, $K$ is the number of memories, and $\mathbf{s}$ is the state of the network. 
The function $F$ maps the similarity (inner product) between the state and memory vectors to a scalar energy value. The choice of $F$ determines the type of memory dynamics and convergence behavior. There are many choices for $F$, but one particularly interesting choice was proposed by [Ramsauer et al. (2020)](https://arxiv.org/abs/2008.02217):
$$
\begin{align*}
E(\mathbf{s}) &= -\texttt{lse}(\beta,\mathbf{X}^{\top}\mathbf{s}) + \frac{1}{2}\mathbf{s}^{\top}\mathbf{s} + \frac{1}{\beta}\log(K)+ \frac{1}{2}M^{2} \\
\end{align*}
$$
where $\mathbf{X}\in\mathbb{R}^{N\times{K}}$ is the matrix of memories, i.e., each memory $\mathbf{m}_{1},\dots,\mathbf{m}_{K}$ consisting of $N$ features is a column of the matrix, $\mathbf{s}$ is the current state of the network, and $\texttt{lse}(\cdot)$ is the log-sum-exp function:
$$
\begin{align*}
\texttt{lse}(\beta,\mathbf{z}) &= \frac{1}{\beta}\log\left(\sum_{i=1}^{K}\exp(\beta\,\mathbf{z}_{i})\right) \\
\end{align*}
$$
and $\beta$ is an inverse temperature parameter that controls the sharpness of the distribution. Finally, $M$ is the largest norm of all the memories, i.e., $M = \max_{i=1,\dots,K}\|\mathbf{m}_{i}\|$. The constants $\frac{1}{\beta}\log(K)$ and $\frac{1}{2}M^2$ ensure the energy remains bounded and comparable across different configurations.

The __vector__ $\mathbf{X}^{\top}\mathbf{s}$ computes the similarity (dot product) between the current state $\mathbf{s}$ and each stored memory, producing a $K$-dimensional vector of similarities. The log-sum-exp function then aggregates these similarities in a smooth, differentiable manner.

<div>
    <center>
        <img src="figs/Fig-Matrix-Vector-Right-Ab-product-NeedToRedrawThis.png" width="580"/>
    </center>
</div>

### Algorithm: Memory retrieval
The user provides a set of memory vectors $\mathbf{X} = \left\{\mathbf{m}_{1}, \mathbf{m}_{2}, \ldots, \mathbf{m}_{K}\right\}$, where $\mathbf{m}_{i} \in \mathbb{R}^{N}$ is a memory vector of size $N$ and $K$ is the number of memory vectors. Further, the user provides an initial _partial memory_ $\mathbf{s}_{\circ} \in \mathbb{R}^{N}$, which is a vector of size $N$, and specifies the _inverse temperature_ $\beta$ of the system.

__Initialize__ the network with the memory matrix $\mathbf{X}$ and inverse temperature $\beta\in\mathbb{R}_{>0}$. Set the current state $\mathbf{s} \gets \mathbf{s}_{\circ}$, initialize the iteration counter $t \gets 1$, maximum iterations $\texttt{maxiter}$, and set convergence flag $\texttt{converged} \gets \texttt{false}$ and tolerance $\epsilon > 0$.

> **Parameter Guidelines**: Common choices are `maxiter = 1000` and $\epsilon$ = `1e-6`. Modern Hopfield networks typically converge within 10–100 iterations, making `maxiter = 1000` a conservative upper bound. The tolerance $\epsilon$ = `1e-6` provides good precision for most applications while avoiding numerical precision issues.

While not $\texttt{converged}$ and $t \leq \texttt{maxiter}$ __do__:
   1. Compute the _current_ similarity vector $\mathbf{z} = \mathbf{X}^{\top}\mathbf{s}$, where each element $z_i = \mathbf{m}_i^{\top}\mathbf{s}$ represents the similarity between the current state and memory $i$.
   2. Compute the _current_ probability vector $\mathbf{p} = \texttt{softmax}(\beta\cdot\mathbf{z})$ where $\texttt{softmax}(\mathbf{u})_i = \frac{\exp(u_i)}{\sum_{j=1}^{K}\exp(u_j)}$.
   3. Compute the _next_ state vector $\mathbf{s}^{\prime} = \mathbf{X}\mathbf{p}$ using the current probability vector $\mathbf{p}$ and the memory matrix $\mathbf{X}$. This step computes a weighted sum of the memory vectors based on the probabilities.
   4. **Check for convergence**: If $\lVert \mathbf{s}^{\prime} - \mathbf{s}\rVert_{2} \leq \epsilon$, then set $\texttt{converged} \gets \texttt{true}$.
      - **Alternative**: If $\lVert \mathbf{p} - \mathbf{p}_{\text{prev}}\rVert_{1} \leq \epsilon_p$, where $\mathbf{p}_{\text{prev}}$ is the probability vector from the previous iteration, $\epsilon_p$ is the convergence tolerance for probabilities (default: $\epsilon_{p}$ = `1e-8`), and $\lVert\star\rVert_{1}$ is the L1-norm, then set $\texttt{converged} \gets \texttt{true}$.
   5. **Update state**: $\mathbf{s} \gets\mathbf{s}^{\prime}$ and increment $t \gets t + 1$.

> **Note**: The softmax function in step 2 is directly related to the log-sum-exp function in the energy formulation. Specifically, the gradient of the LSE with respect to $\mathbf{s}$ yields the softmax-weighted combination of memories used in the update rule.

### Convergence

Modern Hopfield networks have even stronger convergence properties than their classical counterparts, making them highly effective for practical applications.

* **Guaranteed Convergence**: Like classical Hopfield networks, modern variants are **guaranteed to converge** to a fixed point. The energy function serves as a Lyapunov function that decreases monotonically with each update until reaching a minimum.
* **Exponential Convergence Rate**: Modern Hopfield networks exhibit **exponential convergence** to stored memories, dramatically faster than the polynomial convergence of classical networks. The softmax operation creates a _winner-take-all_ dynamic that rapidly identifies and converges to the most similar stored pattern.

Let's discuss convergence of modern Hopfield networks in practice.

> **Convergence** 
>
> In practice, modern Hopfield networks converge quickly: in the best case, convergence occurs in 1–5 iterations. However, in the worst case, it may take 100–200 iterations, especially if the initial state is far from any stored memory or if the memories are highly correlated.
>
> **Factors Affecting Convergence**:
> - **Inverse temperature β**: Higher β leads to faster convergence but may reduce the basin of attraction, i.e., the range of initial states that converge to a given memory
> - **Memory separation**: Well-separated memories in the feature space converge faster, i.e, they are easier to distinguish
> - **Initialization quality**: Starting closer to any stored pattern leads to faster convergence

The exponential convergence rate, combined with increased storage capacity and continuous memory representations, makes modern Hopfield networks significantly more practical than classical variants for real-world applications, especially in high-dimensional continuous data scenarios.

Let's look at an example to illustrate these concepts.

> __Example:__
> 
> [▶ Analyze a modern Hopfield Network](CHEME-5800-L15c-Example-ModernHopfieldNetwork-Fall-2025.ipynb). In this example, we analyze an example of a modern Hopfield Network to understand how it encodes and retrieves gray scale (continuous) patterns. This example builds on the concepts introduced in the previous lecture and demonstrates the application of modern Hopfield networks in continuous associative memory tasks.

Background reading for this lecture (and the associated lab) can be found from the following sources:
* [Krotov, D., & Hopfield, J.J. (2016). Dense Associative Memory for Pattern Recognition. ArXiv, abs/1606.01164.](https://arxiv.org/abs/1606.01164)
* [Demircigil, M., Heusel, J., Löwe, M., Upgang, S., & Vermet, F. (2017). On a Model of Associative Memory with Huge Storage Capacity. Journal of Statistical Physics, 168, 288 - 299.](https://arxiv.org/abs/1702.01929)
* [Ramsauer, H., Schafl, B., Lehner, J., Seidl, P., Widrich, M., Gruber, L., Holzleitner, M., Pavlovi'c, M., Sandve, G.K., Greiff, V., Kreil, D.P., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2020). Hopfield Networks is All You Need. ArXiv, abs/2008.02217.](https://arxiv.org/abs/2008.02217)
* [Krotov, D., & Hopfield, J.J. (2020). Large Associative Memory Problem in Neurobiology and Machine Learning. ArXiv, abs/2008.06996.](https://arxiv.org/abs/2008.06996)

The following blog post is also helpful: [Hopfield Networks is All You Need Blog, GitHub.io](https://ml-jku.github.io/hopfield-layers/)

## Summary

In this lecture, we explored modern Hopfield networks as a powerful generalization of classical associative memory systems:

> __Key takeaways:__
>
> * **Modern Hopfield Energy Function**: Modern Hopfield networks use the log-sum-exp energy function to store continuous patterns rather than binary states. The inverse temperature parameter $\beta$ controls the sharpness of memory retrieval, with higher values producing sharper, more decisive pattern selection through the softmax operation.
> * **Exponential Storage and Convergence**: Modern Hopfield networks achieve exponentially larger storage capacity than classical networks and exhibit exponential convergence (typically 1–5 iterations) through softmax-weighted updates. This dramatic improvement over the classical $0.138N$ storage limit and polynomial convergence makes them practical for high-dimensional continuous data.
> * **Connection to Attention Mechanisms**: The modern Hopfield update rule is mathematically equivalent to single-head attention in transformer architectures. This connection reveals that attention mechanisms can be understood as associative memory retrieval, bridging classical neural network theory with contemporary deep learning.


Modern Hopfield networks provide both theoretical insight into attention mechanisms and practical tools for continuous associative memory tasks in high-dimensional spaces.
___
