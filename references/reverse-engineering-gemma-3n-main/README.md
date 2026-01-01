# Reverse Engineering Gemma 3n: Google's New Edge-Optimized Language Model

On May 20, 2025, as part of Google I/O, Google DeepMind announced Gemma-3n, a new "open" model optimized for running locally on mobile devices. They uploaded weights to Hugging Face—but in the LiteRT MediaPipe .task format—basically a compiled "binary" representation of the model, but no source code.

The announcement post (https://ai.google.dev/gemma/docs/gemma-3n) only references one piece of research used in that model—MatFormer (https://arxiv.org/pdf/2310.07707) which allows the model to run on a subset of weights (there are two released models, E4B and E2B, where like a matryoshka nesting doll, the E2B weights are fully contained within E4B).

The post also mentions "per layer embeddings" which seems to allow them to leave almost half of the parameters of the model on flash memory saving precious RAM usage—but it doesn't explain how it works.

How does this new model work?

There seems to be a lot of little tricks that I haven't really seen elsewhere— and I think personally I've learned a decent amount from trying to figure it out, but an ulterior aim of this is in the vein of "the best way to learn something is to post something wrong on the internet"— hopefully someone with a better idea of what's going on can do a better job. 

Ultimately it'd be nice if we could have some readable open-source implementation which can be ported to environments like `llama.cpp` or Huggingface Transformers or MLX. I've asked Claude to make sense of some of the opcodes and to draft a pytorch implementation (included in repo)— but it's a far cry from something that is actually runnable. 

## First Steps

The first thing I did after downloading the `.task` file (from https://huggingface.co/google/gemma-3n-E4B-it-litert-preview/tree/main) was to open it up in a hex editor. I noticed that it started with the letters "PK", the initials of Phil Katz, the inventor of the .zip file format, and renamed the `.task` to `.zip` to see what was inside.

This unpacked into several files:
- TF_LITE_EMBEDDER (271.6MB)
- TF_LITE_PER_LAYER_EMBEDDER (1.1GB)
- TF_LITE_PREFILL_DECODE (1.59GB)
- TF_LITE_VISION_ADAPTER (17.8MB)
- TF_LITE_VISION_ENCODER (152.7MB)
- TOKENIZER_MODEL (4.7MB)
- METADATA (56 bytes)

So the files which are TF_LITE are probably tflite files, which are flatbuffer files which combine the weights of the model, as well as a graph representation of low level operations used to run the model. In that way it's closer to ONNX than it is to safetensors, pth, or ggml files.

I found this library by Zhenhua Wang (https://github.com/zhenhuaw-me/tflite) for parsing tflite files and extracting out the opcodes. I asked Gemini to write some code to use that library to extract out the operations and tensors in each file.

## Tokenization & Embedding

- TF_LITE_EMBEDDER
  - 28 tensors
  - 19 operations

The opcodes come out in a format like this:

```
➡️ Operator 1: SLICE (Opcode Index: 1, Builtin Code Value: 65)
```

```
Inputs (Tensor Indices & Details):
  0: Index 8 (Name: '/dimension_size', Shape: [2])
  1: Index 1 (Name: 'arith.constant', Shape: [1])
  2: Index 2 (Name: 'arith.constant1', Shape: [1])

Outputs (Tensor Indices & Details):
  0: Index 9 (Name: '/dimension_size1', Shape: [1])
```

I simply copy and paste that input into Claude or Gemini and ask it to convert into equivalent PyTorch code to make it more understandable.

This appears to be a relatively straightforward embedding model which maps the token_ids of a vocabulary which is sized at 262144 tokens—which appears to be the same number of tokens as regular Gemma 3 (https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)—to embedding vectors of size 2048.

```python
class Gemma3nEmbedder(nn.Module):
    def __init__(self, vocab_size=262144, embedding_dim=2048,
                 scale_factor=0.00848388671875, dtype=torch.float32):
        super(Gemma3nEmbedder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.scale_factor = scale_factor
        self.embedding = nn.Embedding(vocab_size, embedding_dim, dtype=dtype)
        
    def forward(self, token_ids):
        embeddings = self.embedding(clipped_token_ids)
        scaled_embeddings = embeddings * self.scale_factor
        return scaled_embeddings
```

I also dumped the 'jit(wrapper)/jit(main)/jit(jax_fn)/embedder.lookup_embedding_table/composite' and 'jit(wrapper)/jit(main)/jit(jax_fn)/GeminiModel.decode_graph/GeminiModel.decode_softmax/transformer.decode_softmax/embedder.decode/composite' tensors, which are stored separately in TF_LITE_EMBEDDER and TF_LITE_PREFILL_DECODE, respectively. They are both exactly identical, which shows that the embedding model and the lm_head have tied weights. This is slightly odd because it seems like a waste to have 268M parameters that are just repeated—and unless MediaPipe/LiteRT have some magic way of determining that these two are identical, this might just be a waste of RAM as well.

## How Gemma3n Halves Inference Memory Requirements: Per Layer Embeddings

- TF_LITE_PER_LAYER_EMBEDDER
  - 86 tensors
  - 48 operators

This is the special trick that is mentioned in the announcement that allows it to save a lot of memory. This isn't guaranteed to be right—but this is my best guess as to why it works:

Each token is initially mapped by the "embedding" to a some vector which encodes some amount of semantic meaning. Over a decade ago with word2vec, researchers demonstrated that these embedding vectors encode a lot of structure about the relationships between concepts and objects—enough to form analogies like "king - man + woman = queen". As we saw earlier, the width of the embeddings is 2048—meaning that there are roughly 2048 different dimensions by which tokens can relate to each other.

This isn't totally right because a language model could probably work "fine" with less than 18 dimensions in its embedding, given that the vocabulary is 2^18. The problem with that is that all the knowledge the model has about the relationships between concepts needs to be stored in the actual layers of the model (likely the FFN, or feed-forward network part of the transformer). Storing those parameters there would be a waste, because those parameters need to be read and computed with on every inference—even if it is completely irrelevant to the current prompt.

So the advantage of keeping information in the embedding is that knowledge can be stored along with certain tokens and then included "on-demand" when they are relevant, rather than all the time.

If we want a language model to be smarter, we might want to give it the ability to represent more concepts by increasing d_model from 2048 to something higher. However, unfortunately this would have the cost of making everything bigger and slower. In a classical GPT-style transformer, the majority of the compute is spent in a matrix multiplication (the FFN) which involves ~(4*d_model)^2 operations, so doubling the number of dimensions would potentially slow things down by 64 times!

Another thing about GPT-style transformers is that they are organized into different layers—where tokens start off as these d_model (2048)-width vectors, and they are progressively morphed into higher level concepts. For example, the first layers may be focused more on things like spelling and capitalization, followed by some layers that focus more on grammar, followed by some layers that focus on meaning, followed by some layers that focus on metaphor.

For a traditional GPT-style transformer, the "embedding" might want to encode all the information associated with a token at each "level"—spelling, capitalization, grammar, and meaning. For things like "meaning"—the model has to learn to take that information and just pass it through the first dozen layers or so without modifying it, because that level of understanding is only useful starting, say, at the 13th layer.

So that's the gist of the idea behind per layer embeddings—the regular embeddings can capture features associated with the model which are relevant starting from the first layer. But per-layer embeddings stores a 256-length vector for each of the 30 layers for each of the 262144 tokens. This gives the opportunity for the model to "look-up" facets of different tokens when they are relevant—rather than having to include them all at the beginning and pass them through like hot-potato until they reach a layer that might find the information useful.

But how does this trick allow you to halve memory consumption? Normal parameters are used for each token, so they all need to be loaded into RAM. But for each token being processed in the autoregressive decoding stage, for the per-layer-embeddings you only need to load 30*256=7,680 (and since these are INT4, this amounts to just under 4KB) per token. 4KB can be loaded relatively quickly on an SSD, and it is a lot less than loading all 2B parameters (~1GB) into RAM.

Why can't this trick be used for the embedding model as well as the per-layer-embeddings? After all, to look up an embedding, you only need to seek to that part of the disk and read 2048 numbers. The answer is that when you are decoding the language model, you need to map the resulting state back into tokens (logits), and that is done (typically) by multiplying it by the transpose of the embedding matrix.

```python
class Gemma3nPerLayerEmbedder(nn.Module):
    def __init__(self, vocab_size=262144, embedding_dim=256, num_layers=30):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Create 30 embedding tables, one for each layer
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, token_ids):
        layer_embeddings = []
        for i, embedding_layer in enumerate(self.embeddings):
            layer_embeddings.append(embedding_layer(token_ids))
        per_layer_embeddings = torch.stack(layer_embeddings, dim=2)
        return per_layer_embeddings
```

Going back a little bit, each layer has 256 numbers in its per-layer-embedding—how does that get mapped back onto the 2048 width residual stream? To figure that out we have to look ahead a little bit and extract a bit from TF_LITE_PREFILL_DECODE.

Basically, each transformer layer has two additional tensors—a `gate` which takes the current value of the hidden state and projects it down into a 256-vector followed by a gelu activation (which is essentially a smoother version of ReLU). That gets multiplied with the per-layer embedding and then another matrix projects it back out to 4096 and it gets added to the residual stream.

Essentially it uses the current value of the residual stream to "decide" which pieces of the per-layer embeddings are relevant (varying from not at all, somewhat, and a lot) and then incorporates those aspects back into the stream.

This all happens at the very end of the layer, so you theoretically have time to scramble to load the embeddings from flash memory starting at the beginning of the layer and hopefully it'll have loaded by the time you need to access it.

```python
class Gemma3nTransformerLayer(nn.Module):
    def __init__(self, d_model=2048, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        # ...
        # Per-layer embeddings
        self.per_layer_gate = nn.Linear(d_model, 256, bias=False)
        self.per_layer_proj = nn.Linear(256, d_model, bias=False)
        self.per_layer_norm = LayerNorm(d_model)
        
    def forward(self, x, position_ids, per_layer_emb):
        # ...
        # Add per-layer embeddings
        gate = F.gelu(self.per_layer_gate(x))
        per_layer_out = self.per_layer_proj(gate * per_layer_emb[:, :, self.layer_idx])
        per_layer_out = self.per_layer_norm(per_layer_out)
        return x + per_layer_out
```

## The Main Text Decoder

- TF_LITE_PREFILL_DECODE
  - 3895 tensors
  - 3009 operators

This is the heart of the language model, and it's also by a considerable margin the most complicated. Just the opcodes from the dump are almost 4MB of pure text, which makes it big enough that it's not feasible to paste it all into Claude or Gemini.

Fortunately, it's quite repetitive—since we can assume there are basically just 30 identical layers stacked together in sequence.

Skimming through the opcode names, the first interesting observation is that a lot of the functions are named e.g. `GeminiModel.decode_graph/GeminiModel.encode`, that is Gemini, not Gemma. This might mean that the Gemma3n model may share some more in common with the mainline Gemini models than the previous Gemma models—which would be neat if true!

The other interesting takeaways from the tensor names are mentions of `altup` and `laurel`.

### LAuReL: Learned Augmented Residual Layers

Laurel seems to refer to LAuReL, "Learned Augmented Residual Layers" https://arxiv.org/abs/2411.07501 from Google research posted on arXiv last November and presented during a workshop at last year's ICML. This seems to be confirmed by https://x.com/GauravML/status/1924917897488658710. The type of Laurel block that appears to be used in Gemma-3n seems to be the "Laurel-LR (Low Rank)" variety.

```python
class LaurelBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.linear_left = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.linear_right = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.norm = RMSNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low-rank transformation
        transformed = self.linear_right(self.linear_left(x))
        normalized = self.norm(transformed)
        return x + normalized
```

Essentially, instead of performing a dense matrix multiplication, where a (2048x2048) matrix would require 4 million parameters, and 4 million multiply-add-carry (MAC) operations, it performs a multiplication with a "low-rank" matrix instead. This effectively means rather than a square matrix, we multiply two rectangular matrices (2048x64) and (64x2048) by our input vector (2048). The end-result is similar (in that we also get a 2048 vector as a result) to that dense matrix multiplication, but we only need 262,000 parameters instead, a 16x reduction in compute and memory.

Rather than using this low rank multiplication as a replacement of another matrix multiplication, it is being used as a "smarter" residual connection. Whereas it's common to have essentially `x' = x + attention(x)`, Gemma3n instead does `x' = (L*R)*x + attention(x)`.

### AltUp: Alternating Updates for Efficient Transformers

The other trick that Gemma3n uses is AltUp (https://arxiv.org/abs/2301.13310). This one was first posted on arXiv over 2 years ago, on January 30, 2023, and presented during NeurIPS 2023.

## Vision

Gemma 3n is multimodal and supports vision as well as audio, though the audio parts of the weights have not been released yet.

### TF_LITE_VISION_ADAPTER

- 29 tensors
- 19 operations

It takes the features from the vision encoder and then transforms them into "soft tokens" which are in the embedding space of the prefill decoder and then can be used to integrate with the prefill decoder. This is pretty much just a matrix multiplication, nothing particularly fancy.

```python
class Gemma3nVisionAdapter(nn.Module):
    def __init__(self,
                 vocab_size=128,
                 hidden_dim=2048,
                 soft_token_len=256,
                 eps=1e-6):
        super().__init__()
        
        self.input_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.soft_embedding_norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.mm_input_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.input_norm_after_projection = nn.LayerNorm(hidden_dim, eps=eps)
    
    def forward(self, soft_tokens, input_indices):
        projected_input = self.input_embedding(input_indices).unsqueeze(1)
        concatenated = torch.cat([soft_tokens, projected_input], dim=1)
        normalized_soft = self.soft_embedding_norm(concatenated)
        projected = self.mm_input_projection(normalized_soft)
        output = self.input_norm_after_projection(projected)
        return output
```

### TF_LITE_VISION_ENCODER

- 2020 tensors
- 1448 operations

One of the first operations uses a tensor named: 'jit(wrapper)/jit(main)/jit(jax_fn)/VisionEncoder/mobilenet/block_group_conv2d_0/Conv_0/conv_general_dilated'. This seems to mean it's based on MobileNet, a family of convolutional neural networks used for computer vision tasks optimized for edge computing.

It appears to include attention and the universal inverted bottleneck so it's probably based on MobileNetV4 published September 2024.


## Next

Like I mentioned at the start, ultimately it'd be nice if we could have some readable open-source implementation which can be ported to environments like `llama.cpp` or Huggingface Transformers or MLX. Also I didn't really pay very much attention to the vision stuff, and I didn't really put that much effort into understanding AltUp. 

