---
title: Quantized SNNs
parent: Implementation
nav_order: 2
---

# Quantized Spiking Neural Networks

{: .highlight }
>Based on the paper:
>
>*Navigating Local Minima in Quantized Spiking Neural Networks*
>
>[(link)](https://arxiv.org/abs/2202.07221)

## Spiking Neuron Model

The leaky integrate-and-fire neuron is described by the discrete-time dynamics:

$$
\begin{align}
u^j_{t+1} &= \beta u^j_t + \sum_i w^{ij} z^i_t - z^j_t \theta \\

z^j_t &=
\begin{cases}
1, \hspace{10pt} \text{if } u^j_t > \theta \\
0,\hspace{10pt} \text{otherwise}
\end{cases}
\end{align}
$$

Where

- $$u^j_t$$ is the hidden state (membrane potential) of neuron $$j$$ at time $$t$$.
- $$\beta$$ is the membrane potential decay rate.
- $$w^{ij}$$ is the synaptic weight between neurons $$i$$ and $$j$$.
- $$\theta$$ is the threshold.

The term $$-z^j_t \theta$$ resets the state by subtracting the threshold each time an output spike $$z^j_t \in \{ 0, 1 \}$$ is generated.

## Periodic LR Schedules

The paper suggests a strategy for adjusting the LR during training. They propose *cosine annealing*, where the LR follows a half cosine curve. The idea is that LR restarts can encourage the model to move from one local minimum to another, allowing the model to explore new regions of the solution space. This is especially important for quantized models that can become stuck in sub-optimal regions.

The LR, $$\eta_t$$, is computed using the following scheduler

$$
\eta_t = \frac{1}{2} \eta \left[ 1 + \cos \left( \frac{\pi t}{T} \right)  \right]
$$

where $$\eta$$ is the initial LR, $$t$$ is the iteration, and $$T$$ is the period of the schedule.
