# Recurrent neuron network

## Main ideas

- Feed forward neuron network does not save any information from an input to another
- Issue when the input is a sequence (video frames, time series ...)
- Idea : add an hidden state which saves a *compressed* version of the information from the previous input (not layers)

**Question** : initialisation of $h_0$ ? Random ? 0 ? Jesus ?

**Question 2** : batch norm and vanishing gradient ?

**Question 3** : BPTT ? 

![](https://miro.medium.com/max/3172/1*mHimR6ok4bAEYhKESwhdrg.png)

## Issue when one tries to learn a RNN :

### Explosing gradient or vanishing gradient

- When the NN is deep, due to the backpropagration formula, the gradient of the cost function tends to explosing or vanishing (cumulative product ...) $\to$ 

#### Reminders (Equations which defined back prop) :

![Equa](http://neuralnetworksanddeeplearning.com/images/tikz21.png)

### Solutions :

- Optimiser algorithms : RMP
- In case of explosing gradients : we can bound the norm of $\nabla C$ using gradient clips.
- Some activation function are less sensitive to this issue : Relu for instance
- A better initialization of weights can avoid partially the issue.