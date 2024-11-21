# NeuralMachineTranslation
Goal: Sentence translation from Cherokee to English  
Assignment 4 of CS 224N of Stanford/ Winter 2021(all the supporting files can be downloaded from https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/) 

## ***Mathematical Description (Training procedure)***

The model uses a *bidirectional LSTM* for source language sentence encoding. Let \( x_{1}, \ldots, x_{m} \) be the embeddings of the source sentence with each \( x_{i} \in \mathbb{R}^{e \times 1} \) for \( i \in [m] \). The last hidden/cell state of the forward directional encoder and the first hidden/cell state of the backward directional encoder are concatenated and projected via linear transforms \( W_{h}, W_{c} \in \mathbb{R}^{h \times 2h} \):
\[
h_{0}^{\textit{dec}} = W_{h}[\overleftarrow{h_{1}^{\textit{enc}}}; \overrightarrow{h_{m}^{\textit{enc}}}], \quad
c_{0}^{\textit{dec}} = W_{c}[\overleftarrow{c_{1}^{\textit{enc}}}; \overrightarrow{c_{m}^{\textit{enc}}}]
\]

The decoder's hidden state dynamics follow:
\[
h_{t}^{\textit{dec}}, c_{t}^{\textit{dec}} = \textit{Decoder}(\overline{y_{t}}, h_{t-1}^{\textit{dec}}, c_{t-1}^{\textit{dec}})
\]

The hidden state \( h_{t}^{\textit{dec}} \) is used to compute multiplicative attention over \( h_{1}^{\textit{enc}}, \ldots, h_{m}^{\textit{enc}} \). Attention scores with respect to the encoders over the time sequence (let's say \( m \)) are computed as follows, for all \( i \in [m] \):
\[
e_{t,i} = \langle h_{t}^{\textit{dec}}, W_{\textit{attProj}} h_{i}^{\textit{enc}} \rangle, \quad W_{\textit{attProj}} \in \mathbb{R}^{h \times 2h}
\]

The *attention probabilities (weights)* are:
\[
\alpha_{t} = \textit{softmax}(e_{t}), \quad e_{t} \in \mathbb{R}^{m \times 1}
\]

The *attention output* is computed as:
\[
a_{t} := \sum_{i=1}^{m} \alpha_{t,i} h_{i}^{\textit{enc}} \in \mathbb{R}^{2h \times 1}
\]

Next, we concatenate the attention output and the hidden state of the decoder to get \( u_{t} = [a_{t}; h_{t}^{\textit{dec}}] \in \mathbb{R}^{3h \times 1} \). Using the *combined output projection matrix* \( W_{u} \in \mathbb{R}^{h \times 3h} \), we compute:
\[
v_{t} = W_{u} u_{t}, \quad o_{t} := \textit{dropout} \big(\tanh(v_{t}) \big)
\]

If \( V_{\textit{tgt}} \) is the vocab size of the target language, we use the *target vocab projection matrix* \( W_{\textit{vocab}} \in \mathbb{R}^{V_{\textit{tgt}} \times h} \) to compute the output probabilities:
\[
P_{t} = \textit{softmax}(W_{\textit{vocab}} o_{t}) \in \mathbb{R}^{V_{\textit{tgt}} \times 1}
\]

On the \( t \)-th step, we look up the embedding of the \( t \)-th subword \( y_{t} \in \mathbb{R}^{e \times 1} \) and concatenate it with \( o_{t-1} \in \mathbb{R}^{h \times 1} \) to get the input for the decoder at the \( t \)-th step:
\[
\overline{y}_{t} := [y_{t}; o_{t-1}] \in \mathbb{R}^{(h+e) \times 1}
\]

Finally, to train the network, we compute the softmax cross-entropy loss between \( P_{t} \) and \( g_{t} \), where \( g_{t} \) is the one-hot vector of the target subword at timestep \( t \). The associated loss at the \( t \)-th decoding step is:
\[
J_{t}(\theta) := \textit{CrossEntropy}(P_{t}, g_{t})
\]

Here, \( \theta \) represents all the trainable parameters of the architecture.

