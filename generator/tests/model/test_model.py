# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# This file has been modified from the original Llama 3 source code.

import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Iterable
from clearml import Logger, Task

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 128
SEQ_LEN = 64
VOCAB_SIZE = 7
NEXT_PROB = .1
INITIAL = 2
    
@dataclass
class ModelArgs:
    vocab_size = VOCAB_SIZE # Added!

    dim: int = 16  # Play to determine the best value
    n_layers: int = 4
    n_heads: int = 4
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    rope_theta: float = 21000

    max_batch_size: int = 128
    max_seq_len: int = 258

    out_channel_sizes: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    strides: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    paddings: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    scaling_factor: int = 2


# Modified inner transformer model defintion
class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)

    Args:
    - dim (int): The input dimension
    - eps (float): A small value to prevent division by zero

    Attributes:
    - eps (float): A small value to prevent division by zero
    - weight (torch.nn.Parameter): The learnable weight parameter
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequencies for the rotary embeddings.

    Args:
    - dim (int): The input dimension
    - end (int): The end value
    - theta (float): The theta value

    Returns:
    - torch.Tensor: The precomputed frequencies of shape (end, dim)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape the frequencies for broadcasting.

    Args:
    - freqs_cis (torch.Tensor): The frequencies
    - x (torch.Tensor): The input tensor

    Returns:
    - torch.Tensor: The reshaped frequencies
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the rotary embeddings to the input tensors.

    Args:
    - xq (torch.Tensor): The query tensor
    - xk (torch.Tensor): The key tensor
    - freqs_cis (torch.Tensor): The frequencies

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: The query and key tensors with the rotary embeddings applied
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    Multi-head attention layer.

    Args:
    - args (ModelArgs): The model arguments

    Attributes:
    - n_heads (int): The number of heads
    - head_dim (int): The dimension of each head
    - wq (nn.Linear): The query weight matrix
    - wk (nn.Linear): The key weight matrix
    - wv (nn.Linear): The value weight matrix
    - wo (nn.Linear): The output weight matrix
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seqlen, seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    Feed-forward layer with Swish-Gated Linear Unit (SwiGLU) activation.

    Args:
    - dim (int): The input dimension
    - hidden_dim (int): The hidden dimension
    - multiple_of (int): The multiple of the hidden dimension

    Attributes:
    - w1 (nn.Linear): The first weight matrix
    - w2 (nn.Linear): The second weight matrix
    - w3 (nn.Linear): The third weight matrix
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.

    Args:
    - layer_id (int): The layer ID
    - args (ModelArgs): The model arguments

    Attributes:
    - n_heads (int): The number of heads
    - dim (int): The input dimension
    - head_dim (int): The dimension of each head
    - attention (Attention): The attention layer
    - feed_forward (FeedForward): The feed-forward layer
    - layer_id (int): The layer ID
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """
    Transformer model with multiple model blocks.

    Args:
    - params (ModelArgs): The model arguments

    Attributes:
    - params (ModelArgs): The model arguments
    - n_layers (int): The number of layers
    - layers (torch.nn.ModuleList): The list of model blocks
    - norm (RMSNorm): The RMSNorm layer
    - freqs_cis (torch.Tensor): The precomputed frequencies
    """
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        # Added : embedding
        self.embedding = torch.nn.Embedding(params.vocab_size, params.dim)
        # -----------------

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

        # Added : final projection (into distribution over vocabulary)
        self.final_proj = torch.nn.Linear(params.dim, params.vocab_size)
        # -----------------


    def forward(self, input_tensor: torch.Tensor, start_pos: int = 0):
        # Added : embedding
        input_tensor = self.embedding(input_tensor) # (B, S) -> (B, S, dim)
        # -----------------
        
        _bsz, seqlen, _ = input_tensor.shape
        h = input_tensor
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=input_tensor.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=input_tensor.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)

        # Added : final projection
        h = self.final_proj(h)
        # -----------------
        
        output = h.float()
        return output

# II. Adjusted trainer

BATCH_NORM_TYPES = (
    torch.nn.BatchNorm1d
    | torch.nn.BatchNorm2d
    | torch.nn.BatchNorm3d
    | torch.nn.SyncBatchNorm
    | torch.nn.LazyBatchNorm1d
    | torch.nn.LazyBatchNorm2d
    | torch.nn.LazyBatchNorm3d
)

def setup_clearml():
    access_key = os.getenv('CLEARML_ACCESS_KEY')
    secret_key = os.getenv('CLEARML_SECRET_KEY')

    Task.set_credentials(
        web_host='https://app.clear.ml',
        api_host='https://api.clear.ml',
        files_host='https://files.clear.ml',
        key=access_key,
        secret=secret_key
    )

class Trainer:
    def __init__(self, lr: float = 2e-4, weight_decay: float = 3e-5,
                 batch_norm_momentum: float | None = 0.002, n_epochs: int = 10,
                 device: str = DEVICE, extra_augmentation: v2.Transform | None = None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_norm_momentum = batch_norm_momentum
        self.n_epochs = n_epochs
        self.device = device
        self.extra_augmentation = extra_augmentation  # TODO: Add reasonable extra augmentation

    def get_optimizer_and_scheduler(
            self, parameters: Iterable[torch.nn.Parameter]
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay, fused=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
        return optimizer, lr_scheduler

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets)

    def evaluate(self, model: torch.nn.Module, test_loader: DataLoader, epoch: int, logger: Logger = None):
        model.eval()
        total_loss = 0.0
        batch_count = 0
        progress_bar = tqdm(enumerate(test_loader), desc="Evaluate")
        for i, (batch, y) in progress_bar:
            batch = batch.to(self.device)
            #batch = transformations.transform_image_to_trainable_form(batch)
            predictions = model(batch[:, :-1])

            #loss = self.compute_loss(predictions, batch[:, 1:])
            loss = F.cross_entropy(predictions.reshape(-1, predictions.size(-1)), batch[:, 1:].reshape(-1))

            total_loss += loss.item()
            batch_count += 1

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
        
        if logger is not None:
            logger.report_scalar(
                title="Validation Loss", series="Inner Transformer Loss", iteration=epoch, value=avg_loss
            )

    def evaluate_accuracy(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
        model.eval()
        sum_acc = 0
        num_examples = 0

        progress_bar = tqdm(enumerate(dataloader), desc="Evaluate")
        for i, (batch, y) in progress_bar:
            batch = batch.to(self.device)
            #batch = transformations.transform_image_to_trainable_form(batch)
            model_out = model(batch[:, :-1])
            y = batch[:, 1:]

            acc = (torch.argmax(model_out, dim=-1) == y).to(torch.float32).sum()
            sum_acc += acc
            num_examples += model_out.shape[0] * model_out.shape[1]

        return sum_acc / num_examples

    def train(self, model: torch.nn.Module, train_loader: DataLoader,
              test_loader: DataLoader, logger: Logger = None):
        model = model.to(self.device)

        if self.batch_norm_momentum is not None:
            # Default torch.nn.BatchNorm2D.momentum is 0.1, but it's often too high.
            for m in model.modules():
                if isinstance(m, BATCH_NORM_TYPES):
                    m.momentum = self.batch_norm_momentum

        optimizer, lr_scheduler = self.get_optimizer_and_scheduler(model.parameters())

        for epoch in range(1, self.n_epochs + 1):
            self.train_epoch(model, train_loader, optimizer, epoch, logger)
            lr_scheduler.step()
            self.evaluate(model, test_loader, epoch, logger)

            # Evaluating accuracy after each epoch
            acc = self.evaluate_accuracy(model, test_loader)
            print(f"{epoch}: Avg eval accuracy {acc}")

    def train_epoch(self, model: torch.nn.Module, train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer, epoch: int, logger: Logger = None):
        model.train()
        total_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(enumerate(train_loader), desc=f"Train epoch {epoch:>3}")
        for i, (batch, y) in progress_bar:
            batch = batch.to(self.device)
            #batch = transformations.transform_image_to_trainable_form(batch)

            optimizer.zero_grad()
            predictions = model(batch[:, :-1])

            #loss = self.compute_loss(predictions, batch[:, 1:])
            loss = F.cross_entropy(predictions.reshape(-1, predictions.size(-1)), batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            
        if logger is not None:
            logger.report_scalar(
                title="Average Epoch Loss", series="Inner Transformer Loss", iteration=epoch, value=avg_loss
            )


# III. Artificial data generation

def generate_recursive(n_first, vocab_size, next_prob):
    assert 0 < vocab_size
    initial = np.random.randint(0, vocab_size, n_first)
    coeffs = np.random.randint(0, vocab_size, n_first)

    return initial, coeffs, vocab_size, next_prob

class SeqGen:
    """
    For generating recurrent sequences with stochastically repeating terms.
    """
    def __init__(self, initial, coeffs, size, next_prob):
        assert len(coeffs) == len(initial)
        self.initial = initial
        self.coeffs = coeffs
        self.size = size
        self.next_prob = next_prob

        self.current = initial

    def __iter__(self):
        return self

    def __next__(self):
        if np.random.random() < self.next_prob:
          new = self.current[-1] + 1
        else:
          new = (self.current @ self.coeffs)

        new %= self.size
        self.current = np.append(self.current, new)[1:]

        return new

    def __key(self):
        return (tuple(self.initial), tuple(self.coeffs), self.size, self.next_prob)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, SeqGen):
            return self.__key() == other.__key()


def generate_dataset(gen_factory, seq_len, num_entries, exclude = []):
    """
    For generating datasets with num_entries elements each
    of length seq_len.

      gen_factory is a procedure that returns
        instance of SeqGen when called.

      seq_len is the length of the sequence to generate.

      num_entries is the number of sequences to generate.

      exclude is the set of sequences that aren't to be used in training
    """
    entries = []
    generators = []
    for e in range(num_entries):
        while True:
          seq_gen = gen_factory()
          if seq_gen in exclude:
              continue

          seq = []
          for s in range(seq_len + 1):
              seq.append(next(seq_gen))

          break

        generators.append(seq_gen)
        entries.append(seq)
    data = torch.tensor(entries, dtype=torch.long)
    # I put data as both x and y due to the way data are handled in Trainer
    return torch.utils.data.TensorDataset(data, data), set(generators)

def example_generator(gen):
    """
      A procedure that returns a representation of
      a single data entrance.
    """
    def example_gen():
        return SeqGen(*gen())
    return example_gen


PERM_EXAMPLE_GENERATOR = example_generator(lambda: generate_recursive(INITIAL, VOCAB_SIZE, NEXT_PROB))

TEST_DATASET, generators = generate_dataset(
    gen_factory=PERM_EXAMPLE_GENERATOR, seq_len=SEQ_LEN, num_entries=1000)
TRAIN_DATASET, _ = generate_dataset(
    gen_factory=PERM_EXAMPLE_GENERATOR, seq_len=SEQ_LEN, num_entries=10000, exclude=generators)


TRAIN_LOADER = torch.utils.data.DataLoader(
    TRAIN_DATASET, batch_size=BATCH_SIZE)
TEST_LOADER = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE)

# IV. Training & evaluation

args = ModelArgs()
trainer = Trainer(n_epochs=70)
# Training the model
model = Transformer(ModelArgs).to(DEVICE)
trainer.train(model, TRAIN_LOADER, TEST_LOADER)