# VLM_blog

## Jointly Training with Image and Text

### VisualBERT ([Li et al. 2019](https://arxiv.org/abs/1908.03557))

注意看 $f_o$ , $f_s$, $f_p$的含义就可以

### SimVLM(Simple Visual Language Model; [Wang et al. 2022](https://arxiv.org/abs/2108.10904))

> **SimVLM** (Simple Visual Language Model; [Wang et al. 2022](https://arxiv.org/abs/2108.10904)) is a simple *prefix language model*, where the prefix sequence is processed with bi-directional attention like BERT, but the main input sequence only has causal attention like [GPT](https://lilianweng.github.io/posts/2022-06-09-vlm/#gpt).

- 在SimVLM中，“前缀语言模型”指的是模型在处理输入序列前，首先使用一个前缀序列来初始化或调整模型的状态。这个前缀序列允许模型在生成主要输入序列的响应之前，先对一些上下文信息进行编码处理。例如，在生成一篇文章的摘要时，前缀可以是“本文摘要：”，然后模型基于这个前缀和文章的内容来生成摘要。

- **Bi-directional attention（双向注意力）**：允许模型在处理每个输入元素时，同时考虑它之前和之后的元素。这种机制类似于BERT模型，适合于理解整个输入序列的全局上下文，因为它可以同时“向前看”和“向后看”。

- **Causal attention（因果注意力）**：只允许模型在处理序列的当前元素时，考虑它之前的元素。这种单向的注意力机制类似于GPT，适合于生成任务，因为它确保了生成的每个元素只依赖于之前的元素，保持了生成过程的因果连贯性。

> According to ablation studies, it is important to have both image-text and text-only data for training. The PrefixLM objective outperforms both [span corruption](https://arxiv.org/abs/1910.10683) and naive LM.

这个结论与[VILA](https://arxiv.org/abs/2312.07533)进行实验后得到的结论是相似的，即纯文本训练数据的加入会提升模型的性能。但VILA叙述的更进一步，作者发现图片-文本对的效果不如使用**图片-文本交错数据**（如一段对图片的叙述）。

![](/Users/wanrenwang/Library/Application%20Support/marktext/images/2024-07-18-22-43-43-image.png)

<center>关于图片-文本对和图片-文本交错数据的区别</center>

使用了ResNet前3层处理的特征图作为Encoder的输入

![](https://lilianweng.github.io/posts/2022-06-09-vlm/SimVLM-arch.png)

### CM3 (Causally-Masked Multimodal Modeling; [Aghajanyan, et al. 2022](https://arxiv.org/abs/2201.07520))

针对超文本的多模态模型，训练数据源是HTML网页。结合了掩码机制和因果机制（因果的解释见上文）。

> Then they are tokenized by [VQVAE-GAN](https://arxiv.org/abs/2012.09841), resulting in 256 tokens per image.

图片的编码方式是经过了VQVAE-GAN，最后得到了256tokens/image的输入。

## Learned Image Embedding as (Frozen) LM Prefix

在这个部分，图像的信息被编码之后，被作为前缀Prefix使用，而LM则或是冻结的或是可以参与微调。

### Frozen([Tsimpoukelli et al. 2021](https://arxiv.org/abs/2106.13884))

将LM参数冻结，训练过程中只训练Vision Encoder（based on NF-ResNet-50 and uses the final output vector of the NF-Resnet **after** the global pooling layer）。后续的消融实验也证明了，在使用预训练LM不进行训练和微调时，效果是最好的。![](/Users/wanrenwang/Library/Application%20Support/marktext/images/2024-07-19-15-52-18-image.png)

> 但这种思路似乎在CLIP出现之后产生了反转，一般的工作都是冻结Vision Encoder（CLIP），因为参数量太大（原因待考）。

使用图片-标题（文本）对数据集进行训练，可以利用到预训练LM拥有的通识知识解决0 shot VQA和few-shot image classification问题。

### ClipCap ([Mokady, Hertz & Hertz, 2021](https://arxiv.org/abs/2111.09734))

> but it needs to be processed by a light mapping network  such that image embedding vectors are translated into the same semantic space as the pre-trained LM.The network $F$ maps CLIP embeddings into a sequence of $k$ embedding vectors, each with the same dimension as a word embedding in GPT2.Increasing the prefix size $k$ helps improve the performance.

正如前述，ClipCap使用了CLIP作为Vision Encoder，但它需要使用一个light mapping network（轻量投射网络？）$F$将得到的image embedding映射到与预训练LM的语义空间相同的语义空间。

*个人理解是通过MLP让image embedding和text embedding在维度上面对齐，然后通过训练发现，这样经过MLP对齐之后就可以达到深层语义对齐的效果。从而反向推导，说出了“such that image embedding vectors are translated into the same semantic space as the pre-trained LM.“，待求证*

ClipCap这种思路与[MobileVLM](https://arxiv.org/pdf/2402.03766)类似，ClipCap将CLIP和LM都冻结，只训练投射器（网络$F$）。但MobileVLM是放开LM参与微调也取得了一定的效果，在ClipCap的实验中，关于这点作者也进行了说明：

> Both CLIP vision encoder and the LM are *frozen* during training and only the mapping network  is learned. They found that when LM is frozen,  should be a transformer, with 8 multi-head self-attention layers with 8 heads each, but when LM can be fine-tuned, a MLP is enough.

这与MobileVLM的实验相吻合，因为其LM就是参与微调的，但其投射器（$F$）使用了Conv，也即一个非Transformer架构的传统神经网络结构，与MLP类似。但MobileVLM的图像是不看做LM的前缀的，在其中进行了一次特征拼接和对齐。

![](/Users/wanrenwang/Library/Application%20Support/marktext/images/2024-07-19-16-07-17-image.png)

> Hence they postulate that “the CLIP space already encapsulates the required information, and adapting it towards specific styles does not contribute to flexibility.”

因ClipCap在image captioning任务达到了当时的SOTA，故其作者团队认为CLIP已经具备了巨量的通识信息，无需进一步专门化以提升其效果或灵活性。强调了CLIP的通用性和强大的信息编码能力。

> The fun fact is - because ClipCap translates CLIP image embeddings into LM space, the processed prefixes can be even interpreted as words.

ClipCap将CLIP生成的image embedding转换到LM的语义空间，作为LM的前缀之后，这些前缀甚至可以被看作是一段由词语组成的话。![](/Users/wanrenwang/Library/Application%20Support/marktext/images/2024-07-19-16-18-30-image.png)

## Text-Image Cross-Attention Fuse Mechanisms

### **VisualGPT** ([Chen et al. 2021](https://arxiv.org/abs/2102.10407))

没整明白

### **VC-GPT** (Visual Conditioned GPT; [Luo et al. 2022](https://arxiv.org/abs/2201.12723))

![](/Users/wanrenwang/Library/Application%20Support/marktext/images/2024-07-20-17-56-30-image.png)

> To avoid catastrophic forgetting, instead of injecting the visual information directly into GPT2, VC-GPT introduces extra cross-attention layers on top of the output of visual encoder and language decoder.

在进入到最终的模态融合步骤前，VC-GPT使用了一个额外的cross-attention层来进行初步的融合，避免灾难性遗忘。

> Then a *self-ensemble* module linearly combines the single model language decoder logits $h^G$ and cross-model vision-language fused module logits $h^{fuse}$ . The self-ensemble module (see “VC-GPT w/o SE” in Fig. 13) is important for the performance.
> 
> $$
> \rm logits = \it W^Gh^G +W^{fuse}h^{fuse}
> $$
> 
> where $W^G$ is a linear projection of the language decoder, initialized by the word embedding matrix of GPT2 and $W^{fuse}$ is a linear projection of the fusion module and initialized randomly.

而后仅采用线性拼接的方法，将单语言模态的输出和视觉-语言交叉融合输出进行拼接。实验证明这个线性拼接层对于性能的提升很大。

*在以往的学习中，text一般都是经过encoder来变为向量，从而和vision向量采取拼接或者对齐的策略进行特征融合。但在VC-GPT中，text使用的是decoder，这将如何进行工作呢？待求证*

### **MERLOT** ([Zellers, et al. 2021](https://arxiv.org/abs/2106.02636))

目标是学习到同时具备空间排序（frame层面）和时序排序（video层面）的能力

> Images are encoded by a learned image encoder and words are encoded using a learned embedding. Then both are encoded together within a joint vision-language transformer.

文本和图片都是先经过一个已经学习好的encoder进行embedding之后，再同时输入一个图像-文本联合Transformer进行特征提取。但具体encoder是什么并未标明，Vision部分有点像ViT。

![](/Users/wanrenwang/Library/Application%20Support/marktext/images/2024-07-20-18-56-51-image.png)

MERLOT经过三个预训练任务，在图中都有体现。注意后两个任务都是只去cls分类标记进行的：

1. *Masked language modeling* (MLM)

2. *Contrastive frame-caption matching*

3. *Temporal reordering*

> Ablation studies showed that it is important to (1) train on videos instead of images, (2) scale up the size and diversity of the training dataset and (3) use diverse objectives to encourage full-stack multimodal reasoning.

为了学习到上述提到的两种能力，消融实验也证明了使用视频来进行训练的效果是最好的（*但是这本来就是个空间和时序能力模型的构建，肯定要用视频最好吧……*），而且scaling law还在发力，同时使用多个任务进行多模态全栈推理似乎对模型能力的激发有帮助？

### **Flamingo** ([Alayrac et al. 2022](https://arxiv.org/abs/2204.14198))

![](/Users/wanrenwang/Library/Application%20Support/marktext/images/2024-07-21-19-17-57-image.png)

> To more efficiently incorporate vision signals, Flamingo adopts a [Perceiver](https://arxiv.org/abs/2103.03206)-based architecture to produce a few hundreds of tokens out of a large number of visual input features and then use cross-attention layers interleaved with the LM layers to fuse visual information into the language decoding process. The training objective is an autoregressive, NLL loss.

Flamingo使用了一个Perceiver-based的结构从大量输入视觉特征中提取出百余个tokens，然后再使用cross-attention层，将视觉token与语言token进行交叉注意力计算，起到融合的作用。训练目标是使用自回归的负对数似然损失，有助于文本生成时更好地考虑图像内容。

```json
> **Perceiver**是一种神经网络架构，旨在处理各种形式的大规模输入数据（如图像、视频、文本等），其设计主要是为了克服Transformer在处理大规模数据时的计算瓶颈。Perceiver架构的主要特点包括：
> 
> - **输入处理**：它首先将大规模输入数据（如图像或视频帧）通过一个固定的小尺寸的潜在空间（latent space）。这一过程可以理解为将高维输入映射到低维空间中，减少计算量。
> - **交叉注意机制**：在Perceiver架构中，输入数据和潜在空间通过交叉注意机制进行交互。这种机制使得模型能够在潜在空间中高效地捕捉和表示输入数据的关键信息。
> - **迭代处理**：这种交互过程可以迭代进行多次，从而逐步增强潜在空间对输入数据的理解和表征。
> 
> 这种架构允许Perceiver处理非常大的输入数据，而不会像传统的Transformer那样在计算上变得不可行。

> **负对数似然损失（Negative Log-Likelihood Loss, NLL）** 是一种常用的损失函数，特别是在自回归语言模型中。选择这种损失函数的原因和其特点包括：
> 
> - **适用于分类任务**：NLL损失函数通常用于分类任务中，在语言模型中，每个时间步的目标是预测下一个词的类别（词汇表中的一个词）。
> - **与概率相关**：NLL损失直接与模型输出的概率相关。对于每个预测，NLL损失计算模型预测的概率与真实标签的对数概率之间的差异。具体公式为： NLL=−i∑​logP(yi​∣xi​) 其中，P(yi​∣xi​) 是模型预测的词 yi​ 的概率，给定输入 xi​。
> - **平滑优化**：NLL损失在概率空间中进行优化，损失值较为平滑，有利于梯度下降算法的收敛。
> - **自回归训练**：对于自回归模型（例如语言模型），NLL损失可以逐步地进行训练，每一步预测依赖于之前的预测，符合生成序列的本质。
```

> To easily handle text with interleaved images, masking in Flamingo is designed such that text token only cross-attends to visual tokens corresponding to the *last* preceding image, largely reducing the number of visual tokens that a certain text token can see. They found this works better than allowing text tokens to attend to all preceding images directly. Text still can attend to all previous images because there is a causal self-attention dependency in the text encoder. This design can deal with an arbitrary number of images in the context.

```json
1. "last preceding image" 是指同时输入进去的两张图片的后一张吗？
解释：
这里的 "last preceding image" 指的是在文本生成过程中，某个特定的文本标记（token）只会与紧邻它之前的最后一张图片（视觉标记）进行交叉注意（cross-attention）。如果有多张图片输入，那么对于每个文本标记，它只会关注在它之前的最后一张图片。例如，如果输入顺序是图像1，图像2，文本，那么这个文本标记只会交叉注意图像2的视觉标记。

2. “They found this works better than allowing text tokens to attend to all preceding images directly.” 这句话要如何理解呢？
解释：
这句话的意思是，在设计模型时，他们发现让文本标记只与最后一张之前的图片进行交叉注意，比让文本标记直接关注所有之前的图片效果更好。
这样做的主要原因可能是：
减少计算复杂度：每个文本标记只需要关注最后一张图片的视觉标记，减少了需要处理的视觉标记数量，从而降低了计算复杂度。
提高注意力的集中性：这样可以让模型在每一步生成文本时更集中地利用最近的图像信息，避免了注意力的分散，提高了生成的准确性和相关性。

3. “Text still can attend to all previous images because there is a causal self-attention dependency in the text encoder.” 这句话的理解
解释：
这句话的意思是，尽管文本标记只与最后一张之前的图片进行交叉注意，但文本编码器内部的因果自注意机制（causal self-attention）允许文本标记间相互注意。这意味着文本标记可以通过注意机制间接地获取到所有之前图片的信息。

具体来说，模型在生成文本时，虽然每个文本标记只直接关注最后一张图片，但由于文本编码器内部的因果自注意机制，之前的文本标记（已经融合了之前图片的信息）可以传递信息给当前的文本标记。因此，模型实际上仍然可以考虑到所有之前的图片信息，只是通过间接的方式。
```

通过**掩码**机制，使得文本标记只与紧邻它的图片进行交叉注意力计算。

*大概使用如下代码进行计算，待验证*

```python
import torch

def create_mask(num_images, num_visual_tokens_per_image, num_text_tokens):
    # 初始化全为零的mask，大小为 (num_text_tokens, total_visual_tokens)
    total_visual_tokens = num_images * num_visual_tokens_per_image
    mask = torch.zeros(num_text_tokens, total_visual_tokens)
    
    # 对于每个文本标记，找到它之前的最后一张图片的视觉标记，并设置mask
    for t in range(num_text_tokens):
        # 找到对应的图片索引
        last_image_idx = (t // num_visual_tokens_per_image)
        
        # 计算该图片在mask中的开始和结束位置
        start_idx = last_image_idx * num_visual_tokens_per_image
        end_idx = start_idx + num_visual_tokens_per_image
        
        # 设置mask，文本标记 t 只关注它之前的最后一张图片的视觉标记
        mask[t, start_idx:end_idx] = 1
        
    return mask

# 示例参数
num_images = 3
num_visual_tokens_per_image = 5
num_text_tokens = 10

# 创建mask
mask = create_mask(num_images, num_visual_tokens_per_image, num_text_tokens)
print(mask)

```

> 解释：
> 
> 1. **输入参数**：
>    
>    - `num_images`：输入的图片数量。
>    - `num_visual_tokens_per_image`：每张图片的视觉标记数量。
>    - `num_text_tokens`：文本标记的数量。
> 
> 2. **掩码创建**：
>    
>    - 初始化一个全为零的掩码，大小为 `(num_text_tokens, total_visual_tokens)`，其中 `total_visual_tokens = num_images * num_visual_tokens_per_image`。
>    - 对于每个文本标记，计算它之前的最后一张图片的视觉标记范围，并在掩码中将对应位置设置为1。
> 
> 3. **应用掩码**：
>    
>    - 在模型的交叉注意层中使用这个掩码，确保每个文本标记只关注它之前的最后一张图片的视觉标记。

*应用部分，待验证*

```python
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, text_tokens, visual_tokens, mask):
        Q = self.query(text_tokens)
        K = self.key(visual_tokens)
        V = self.value(visual_tokens)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 应用掩码，只保留对应位置的注意力
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算注意力输出
        attention_output = torch.matmul(attention_weights, V)
        return attention_output

# 示例输入
text_tokens = torch.randn(num_text_tokens, d_model)
visual_tokens = torch.randn(num_images * num_visual_tokens_per_image, d_model)

# 创建模型和掩码
cross_attention = CrossAttention(d_model)
mask = create_mask(num_images, num_visual_tokens_per_image, num_text_tokens)

# 前向传播
output = cross_attention(text_tokens, visual_tokens, mask)
print(output)
```

> They scraped 43 million webpages from the Internet, named MultiModal MassiveWeb (M3W) dataset, containing text with interleaved images.

Flamingo还创建了一个M3W数据集，里面全是文本-图片交错数据集，类似于VILA中提到的MMC4.

> - A function  $\phi: [1:L] \rightarrow [0:N]$ is computed to track the text and image interleaving order, which assigns to each text position the index of the last image/video appearing before this position; 0 if no preceding visual data.

数据处理思路不错，推荐看下原文。下面重点解释一下第三个部分，也即上述的原文。

具体来说，这个函数$\phi$用来跟踪文本和图像/视频的交错顺序，为每个文本位置分配它之前出现的最后一个图像/视频的索引。如果在某个文本位置之前没有出现过图像/视频，则分配为0。

- **输入参数**：
  
  - L：文本的总长度（文本标记的数量）。
  - N：图像/视频的总数量。

- **函数定义**：
  
  - $\phi$:[1:L]→[0:N]
  - 这个函数将文本位置映射到之前出现的最后一个图像/视频的索引。如果在某个文本位置之前没有出现过图像/视频，则返回0。

实现的实例代码大概是：

```python
def compute_phi(sequence):
    """
    计算函数 φ: [1:L] -> [0:N]
    :param sequence: 包含文本和图像/视频的交错序列，其中图像/视频用 "IMG" 表示，文本用 "TEXT" 表示
    :return: 一个列表，表示每个文本位置的 φ 值
    """
    phi = []
    last_image_index = 0
    
    for idx, item in enumerate(sequence):
        if item == "IMG":
            last_image_index += 1
        elif item == "TEXT":
            phi.append(last_image_index)
    
    return phi

# 示例序列
sequence = ["IMG", "TEXT", "TEXT", "IMG", "TEXT", "IMG", "TEXT", "TEXT"]
phi = compute_phi(sequence)
print(phi)  # 输出: [1, 1, 2, 3, 3]

```

- **输入序列**：`sequence` 是一个包含文本和图像/视频交错的列表。这里用 "IMG" 表示图像/视频，用 "TEXT" 表示文本。
- **初始化**：`phi` 列表用于存储每个文本位置的 ϕ 值。`last_image_index` 初始化为0，用于跟踪最后一个出现的图像/视频的索引。
- **遍历序列**：遍历输入序列，对于每个元素：
  - 如果是 "IMG"，则更新 `last_image_index`。
  - 如果是 "TEXT"，则将当前的 `last_image_index` 添加到 `phi` 列表中。

> In practice, instead of round-robin between datasets, they actually sample one batch from each dataset and apply a weighted sum of these gradients in each update.

**在实际操作中，他们不是采用数据集的轮循，而是从每个数据集中采样一个批次，并在每次更新中应用这些梯度的加权和。**

- **轮循（Round-robin）**：指的是依次从每个数据集中取批次进行训练，例如，先从数据集1取一个批次进行训练，然后从数据集2取一个批次进行训练，以此类推。这种方法可能会在某些情况下不太稳定，因为每个数据集的特点不同，直接轮循可能会导致梯度更新的不平衡。

- **每次采样一个批次**：从每个数据集中同时采样一个批次（即在每次训练迭代中从所有数据集中取样），然后计算每个批次的梯度。

- **应用梯度的加权和**：将每个批次的梯度乘以相应的数据集权重，然后求和得到一个总的梯度，最后用这个总的梯度来更新模型参数。这种方法可以稳定训练过程，因为它减少了不同数据集之间的梯度方差。

**为什么这种方法更好？**

1. **稳定性**：每次更新时都考虑所有数据集的梯度，有助于减少梯度的方差，防止训练过程中的不稳定性。
2. **有效性**：通过加权和的方法，可以更加细致地控制每个数据集对模型训练的贡献，确保模型在各个数据集上的表现都能得到优化。
3. **效率**：相比于轮循方法，这种方法更能充分利用每个数据集的信息，提高训练效率。

>  Gradient accumulation across different heterogeneous datasets can be viewed as a mean to stabilize training, as it reduces the gradient variance between each update.

**梯度计算和累积**：基于总的损失值计算梯度，并进行梯度累积。梯度累积有助于稳定训练过程，因为它可以减少每次更新之间的梯度方差。这种做法尤其在处理异构数据集时有效，因为不同数据集的梯度可能有较大的差异。

- **梯度累积**：通常指的是在多个小批次（mini-batches）上计算梯度，并在真正更新模型参数之前累积这些梯度。这里特指在来自不同数据集的批次上进行梯度累积。

- **梯度方差**：是指每次计算的梯度之间的变化幅度。如果梯度方差大，每次更新可能会波动很大，从而影响训练的稳定性。

- **减少梯度方差**：通过在多个数据集上的批次累积梯度，可以平均化每个批次的影响，从而减少每次更新之间的梯度波动。

采取批次计算损失加权和+梯度累计伪代码

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 假设我们有一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# 初始化模型和优化器
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 示例损失函数
criterion = nn.NLLLoss()

# 初始权重
weights = [0.3, 0.3, 0.4]

# 假设我们每累积3个批次的梯度再进行一次更新
accumulation_steps = 3

# 训练循环
for epoch in range(num_epochs):
    # 初始化梯度累积变量
    accumulated_gradients = [torch.zeros_like(param) for param in model.parameters()]

    for step in range(total_steps):
        # 从每个数据集采样一个批次
        batches = [sample_batch_from_dataset(D) for D in [D1, D2, D3]]

        # 计算每个批次的损失和梯度
        for i, batch in enumerate(batches):
            outputs = model(batch['input'])
            loss = criterion(outputs, batch['target'])
            optimizer.zero_grad()
            loss.backward()

            # 获取并累积梯度
            for param, acc_grad in zip(model.parameters(), accumulated_gradients):
                acc_grad += param.grad * weights[i]

        # 每 accumulation_steps 步进行一次参数更新
        if (step + 1) % accumulation_steps == 0:
            for param, acc_grad in zip(model.parameters(), accumulated_gradients):
                param.grad = acc_grad / accumulation_steps  # 平均化累积的梯度
            optimizer.step()

            # 清空累积梯度
            accumulated_gradients = [torch.zeros_like(param) for param in model.parameters()]

    print(f'Epoch [{epoch+1}/{num_epochs}] complete')

```

> - **初始化模型和优化器**：定义一个简单的线性模型，并使用Adam优化器。
> - **初始权重**：设定每个数据集的初始权重 w1, w2, 和 w3。
> - **梯度累积变量**：初始化一个列表 `accumulated_gradients` 来存储累积的梯度。
> - **采样批次并计算梯度**：对每个数据集的批次分别计算梯度，并将这些梯度累积到 `accumulated_gradients` 中，乘以相应的权重。
> - **梯度累积和参数更新**：每累积 `accumulation_steps` 个批次后，使用累积的梯度更新模型参数。注意在更新参数前对累积的梯度进行平均化。
> - **清空累积梯度**：在参数更新后，清空累积梯度，准备下一轮的累积。

### **CoCa** (Contrastive Captioner; [Yu & Wang et al., 2022](https://arxiv.org/abs/2205.01917))

![](/Users/wanrenwang/Library/Application%20Support/marktext/images/2024-07-21-19-18-34-image.png)

> **CoCa** (Contrastive Captioner; [Yu & Wang et al., 2022](https://arxiv.org/abs/2205.01917)) captures both the merits of contrastive learning and image-to-caption generation. It is a model jointly trained with contrastive loss on CLIP-style representation and generative loss on image captioning, achieving SoTA zero-shot transfer on a variety of multi-modal evaluation tasks.

CoCa同时在对比和生成任务上进行训练。对比学习方面，使用了类似CLIP的对比学习策略进行训练。

> **CLIP训练方法**![](https://api2.mubu.com/v3/document_image/90e9eb6b-0858-4cbc-af93-f38bf9f0f8da-1272814.jpg)  
> 
> - 使用对比学习进行训练  
>   
>   - 构建正负图像-文本对，如上图左侧，对角线的就是正例，剩下的都是负例  
> 
> - 模型在训练时只需要判断出正例和负例即可  
> 
> - 使用“构建好的prompt”进行测试，得到的得分最高的就是那一类  
>   - 测试图片是随机的，可是用来构建prompt的词语也可以是随机的，不像传统方法是固定的
> 
> - 伪代码例  
>   
>   - 参数设定
>   
>   - ![](https://api2.mubu.com/v3/document_image/3fba403f-e1c2-46c5-844e-086226b45e67-1272814.jpg)  
>   
>   - 特征提取![](https://api2.mubu.com/v3/document_image/d982841b-5731-42f0-873a-eb7bd1187516-1272814.jpg)
>   
>   - 归一化![](https://api2.mubu.com/v3/document_image/d6ccf20c-8c66-485e-a343-d6fed2e6ad23-1272814.jpg)
>     
>     - 当中的投射层是为了学习特征如何从单模态变到多模态的  
>     
>     - 也就是作者在注释中说的“合并的多模态特征”  
>     
>     - 投射完之后进行L2归一化
>   
>   - 计算分类使用的logits![](https://api2.mubu.com/v3/document_image/0ae0414b-c37c-40da-ad5d-333d6ae9c661-1272814.jpg)
>     
>     - 计算余弦相似度
>   
>   - 损失函数计算![](https://api2.mubu.com/v3/document_image/d4596eac-50c9-4795-9475-6c8926cead99-1272814.jpg)
>     
>     - 和ground truth进行交叉熵函数计算  
>     
>     - 因为所有在对角线上的元素是正样本，所以使用np.arange(n)来创建ground truth？

值得注意的是，

> CoCa is pretrained from *scratch*, using web-scale alt-text data [ALIGN](https://lilianweng.github.io/posts/2022-06-09-vlm/#pair-image-text-datasets) and annotated images by treating all labels as texts in [JTB-3B](https://lilianweng.github.io/posts/2022-06-09-vlm/#pair-image-text-datasets).

CoCa是从零开始训练的，且使用了较大的数据。

> $L_{cap}$：*Encoder-decoder captioning* has the decoder predict the caption based on the latent encoded features from the image encoder, by optimizing an autoregressive loss. The text decoder is decoupled into two components, *unimodal* and *multimodal*; a good balance is to split the decoder by half for these two components:
> 
> - The bottom unimodal component encodes the input text with causally-masked self-attention.
> - The top multimodal component applies both causally-masked self-attention and cross-attention to the output of the vision encoder.

CoCa使用了两个损失函数，$L_{con}$和$L_{cap}$，分别代表对比损失和caption损失。在这里重点看一下$L_{cap}$。可以观察模型图看到，有两个text decoder，下方的单模态decoder负责将输入编码，采用因果掩码注意力*和GPT一样？*。上方的多模态decoder与vision encoder的输出进行cross- attention计算，实现模态融合。然后再使用因果掩码注意力输出对caption的预测，根据预测结果对$L_{cap}$进行计算。![](/Users/wanrenwang/Library/Application%20Support/marktext/images/2024-07-21-19-40-42-image.png)

> They use task-specific attention pooling, or attention pooler, as a natural task adapter, as they found that a single pooled image embedding helps visual recognition tasks (e.g. ImageNet classification), while a more fine-grained embedding helps multimodal understanding tasks (e.g. VQA). A pooler is a single multi-head attention layer with $n_{query}$ learnable queries (note that$\bf X \in \mathbb{R}^{\it L \times d}$,$\bf W^{\it q} \in \mathbb{R}^{\it d \times d_q}$, and $d_k=d_q$), with the encoder output as both keys and values. CoCa uses attentional poolers in pretraining for generative loss $ n_{query}=256$ and contrastive loss $ n_{query}=1$ . This enables the model to obtain strong performance as a *frozen* encoder where we only learn a new pooler to aggregate features.

**注意力池化(attention pooling**是一种将输入序列的表示（embeddings）聚合成单一表示的方法。不同任务对表示的要求不同：

- **视觉识别任务**（如ImageNet分类）通常需要一个单一的、整体的图像嵌入。
- **多模态理解任务**（如视觉问答VQA）需要更细粒度的嵌入来捕捉图像中的细节信息。

一个注意力池化器（Attention Pooler）是一个单一的多头注意力层，它通过$n_{query}$个可学习的查询（queries）来聚合输入表示。

![](/Users/wanrenwang/Library/Application%20Support/marktext/images/2024-07-21-20-03-25-image.png)

**池化结果**：最终的注意力池化结果Z的维度为$\mathbb{R}^{\it n_{query} \times d_v}$，表示为$n_{query}$个聚合后的特征表示，每个特征的维度为 dv​。

CoCa模型在预训练时使用了不同的注意力池化器，具体来说：

- **生成损失**（generative loss）使用$n_{query}=256$，以获得细粒度的嵌入，适用于多模态理解任务。
- **对比损失**（contrastive loss）使用$n_{query}=1$，以获得单一的图像嵌入，适用于视觉识别任务。

> 注意力池化与1*1 conv的区别与联系
> 
> - **维度变化**：
>   
>   - **1x1卷积**：只改变特征维度，不改变空间维度。
>   - **注意力池化**：输出的特征表示数量由查询向量的数量 nquery​ 决定，不是固定的空间维度。
> 
> - **计算方式**：
>   
>   - **1x1卷积**：线性变换（权重矩阵的乘积）。
>   - **注意力池化**：通过注意力机制计算加权和。
> 
> - **应用场景**：
>   
>   - **1x1卷积**：常用于调整特征维度、降低计算复杂度或作为特征融合的一部分。
>   - **注意力池化**：用于根据任务需求对输入表示进行聚合，特别适用于需要从全局特征中提取信息的任务。

## No Traning

### **MAGiC** (iMAge-Guided text generatIon with CLIP; [Su et al. 2022](https://arxiv.org/abs/2205.02655))

公式过于牛逼，看不懂

### PICa (Prompts GPT-3 via the use of Image Captions; [Yang et al. 2021](https://arxiv.org/abs/2109.05014))

> first converts the images into captions or tags and then uses few-shot examples to prompt GPT3 to provide answers. Image captions or tags are extracted by some existing models (e.g. [VinVL](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VinVL_Revisiting_Visual_Representations_in_Vision-Language_Models_CVPR_2021_paper.html)) or Azure Tagging API. And GPT3 is considered as an unstructured, implicit knowledge base.

被提出是用来解决VQA问题。使用现有的模型或API生成对图片的caption或者tag，然后将得到的captions组成few-shot prompt输入到被默认为是终极知识库的GPT3中，来得到问题的最终答案。![](https://lilianweng.github.io/posts/2022-06-09-vlm/PICa-fewshot.png)

> PICa explored two ways to improve few-shot examples to achieve better results:
> 
> - In-context examples are selected based on how *similar* they are to the question using CLIP embedding.
> - *Multi-query ensembling* is to prompt the model multiple times to get multiple answers and the one with highest logprob is selected.



*CLIP嵌入可以用于在图像和文本之间建立相关性，以便在知识问答（VQA）任务中选择最相关的上下文示例。使用CLIP嵌入的步骤应该如下，待考证：*

1. **图像描述生成**：
   使用现有的模型（如VinVL）或Azure Tagging API，将图像转换为文字描述（captions）或标签（tags）。

2. **问题和描述的嵌入生成**：
   使用CLIP模型，将问题和所有图像描述转化为嵌入向量。

3. **计算相似性**：
   计算问题嵌入与每个图像描述嵌入之间的相似性，选择与问题最相似的描述作为上下文示例。

4. **提示GPT-3**：
   使用选定的上下文示例来提示GPT-3生成答案。

### **Socratic Models** (SM) ([Zeng et al. 2022](https://arxiv.org/abs/2204.00598))

> Here language is considered as the intermediate representation by which different models can exchange information. The key idea is to use *multi-model multimodal prompting*, in which output of a non-language model is inserted into a language prompt and then it is used for LM for reasoning.

在SM中，语言被作为一种**沟通各个模型间的中间信息**，个人理解是将LM作为大脑，指挥其他所有模态的模型进行工作。

![](/Users/wanrenwang/Library/Application%20Support/marktext/images/2024-07-21-20-25-18-image.png)

> SM framework is very flexible and can be used on a lot more complicated tasks other than image captions. For example, the egocentric perception (User inputs + VLM + LM + ALM) task is to take as inputs egocentric videos to (1) summarize content; (2) answer free-form reasoning questions; (3) and do forecasting.

SM非常适合用于自拍视频的探索，可能也是因为思维链的缘故。

## Datasets & Evaluation Tasks

看原文吧。