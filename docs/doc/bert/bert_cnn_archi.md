# BERT + CNN

## 1) CNN and NLP





## 2) Encoder (from Transformers) and BERT

*Attention is all you need !*

Transformers is based on two ideas : Attention function, and encoder/decoder structure

The BERT (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers) is an encoder stacking.

#### Attention mechanism 

- database point of view : the attention function mimics the query, looking for a key in our database (which is associated to a value).

- machine point of view : the goal is to modify the embedding to add context to the word embedded $c_k$

  Knowing the query (a word), what is his similarity with all the keys (N word from the sentence) :

  ![attention_function](C:\Users\wenceslas\Documents\cours\ENSAE\2A\Normal\statapp\doc\attention_function.png)
  
  
  
  For each key $c_i$ (embedded word dimension (vector) of $d$) into our sentence, we compute the similarity $r_{k\rightarrow i}$ to our query $c_k$ as :
  
  $$r_{k\rightarrow i} = f(c_k, c_i) = c_k^TW_{c_k}W_{c_i}c_{i}$$ 
  
  $W$ here is a mapping matrix, used to project our query into a new space to allow us to compare query/key (learn by the model).
  
  Then we compute attention scores $$r_{k\rightarrow i}= \frac{\exp{r_{k\rightarrow i}}}{\sum\exp{r_{k\rightarrow j}}}$$ (softmax function) for each key 1 to N.
  
  Finally, we multiply our scalar $r_{k\rightarrow i}$ to his embedded representation $c_i$ (with an $W_{c_i}$ different from the one of similarity function). We sum them to get our new $c_k$ embedding noted $\tilde{c_k}$. **And so, this embedding take into account the context of the sentence.**

In practice, we compute the attention function on a set of queries simultaneously packed into a matrix $Q$. The keys $K$ and the values $V$ are also stacked, and so we have this : 

$$Attention(QW^Q,KW^K,VW^V)=softmax(QW^Q(KW^K)^T)VW^V$$



#### Multi-head attention

We are performing attention function $h$ times  (with different $W$ initialisation) **to capture different attention** and we concat all output :

$$MutiliHead(Q, K, V) = Concat(head_1, ..., head_i, ..., head_h)W^0$$ 

avec

$$head_i= Attention(QW_i^Q,KW_i^K,VW_i^V)$$



#### Positional Encoding

We add to our embedded word the signal of his position in the sentence (sin/cos).



#### Advantages

- Neither gradient explosion nor gradient vanishing (layer normalisation after multi-head and feed-forward)
- Parallels computation (no recurrence)
- Reduce complexity comparing to RNN, LSTM, CNN



## 3) Merging BERT + CNN

