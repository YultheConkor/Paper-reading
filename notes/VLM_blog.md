# VLM_blog

## VisualBERT

注意看$f_o$, $f_s$, $f_p$的含义就可以

## SimVLM

> **SimVLM** (Simple Visual Language Model; [Wang et al. 2022](https://arxiv.org/abs/2108.10904)) is a simple *prefix language model*, where the prefix sequence is processed with bi-directional attention like BERT, but the main input sequence only has causal attention like [GPT](https://lilianweng.github.io/posts/2022-06-09-vlm/#gpt).

- 在SimVLM（简单视觉语言模型）中，“前缀语言模型”指的是模型在处理输入序列前，首先使用一个前缀序列来初始化或调整模型的状态。这个前缀序列允许模型在生成主要输入序列的响应之前，先对一些上下文信息进行编码处理。例如，在生成一篇文章的摘要时，前缀可以是“本文摘要：”，然后模型基于这个前缀和文章的内容来生成摘要。

- **Bi-directional attention（双向注意力）**：允许模型在处理每个输入元素时，同时考虑它之前和之后的元素。这种机制类似于BERT模型，适合于理解整个输入序列的全局上下文，因为它可以同时“向前看”和“向后看”。

- **Causal attention（因果注意力）**：只允许模型在处理序列的当前元素时，考虑它之前的元素。这种单向的注意力机制类似于GPT，适合于生成任务，因为它确保了生成的每个元素只依赖于之前的元素，保持了生成过程的因果连贯性。