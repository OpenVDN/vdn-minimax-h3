"""The Ulysses runtime: rank layout, the six collectives and the Triton pack kernel.

Inference-only sequence parallelism for the H3 hybrid transformer (the forwards that
use it are in ulysses.py). The block residual stream is sharded by packed-sequence row;
inside attention one uneven all-to-all sends processed q/k/v/gate to the softmax ranks
and raw q/k/v/beta plus the low-rank gate hidden to the linear ranks, and one reverse
all-to-all brings every head's output back to its sequence owner. No KV cache, stale
state, token dropping or approximate communication.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import triton
import triton.language as tl


def sequence_splits(sequence_length: int, world_size: int) -> tuple[int, ...]:
    """Contiguous, nearly-even shards with padding confined to the last rank.

    The first ``world_size - 1`` ranks get ceil(S/P) rows and the last rank gets
    the remainder.  H3 has ~100k rows, so the imbalance is at most P-1 rows.  Keeping
    the short shard last lets the final all-gather expose its valid rows as one prefix
    without allocating another full residual-stream copy.
    """
    if sequence_length <= 0 or world_size <= 0:
        raise ValueError("sequence_length and world_size must be positive")
    if world_size == 1:
        return (sequence_length,)
    chunk = math.ceil(sequence_length / world_size)
    last = sequence_length - chunk * (world_size - 1)
    if last <= 0:
        # This only occurs for toy S << P^2.  Keep the helper generally useful; the
        # runtime itself rejects empty shards because NCCL all-to-all does not buy
        # anything for such a sequence.
        base, extra = divmod(sequence_length, world_size)
        return tuple(base + (rank < extra) for rank in range(world_size))
    return (chunk,) * (world_size - 1) + (last,)


def balanced_splits(total: int, parts: int) -> tuple[int, ...]:
    """Split ``total`` contiguously with a maximum imbalance of one.

    Unlike :func:``sequence_splits``, every shard is kept as even as possible.  Head
    shards are long-lived compute assignments, so putting all of the remainder on the
    last rank would directly create a straggler.
    """
    if total <= 0 or parts <= 0 or parts > total:
        raise ValueError(f"need 0 < parts <= total; got total={total}, parts={parts}")
    base, extra = divmod(total, parts)
    return tuple(base + (rank < extra) for rank in range(parts))


@triton.jit
def _pack_branch_target_kernel(
    Q,
    K,
    V,
    SCALAR,
    SHARED,
    OUT,
    ROWS,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FIRST_HEAD: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    SHARED_WIDTH: tl.constexpr,
    LINEAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack one destination directly from source fields into its A2A send segment."""
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    per_head = 3 * HEAD_DIM + 1
    head_payload = LOCAL_HEADS * per_head
    if LINEAR:
        row_width = head_payload + SHARED_WIDTH
    else:
        row_width = head_payload
    mask = (row < ROWS) & (cols < row_width)
    in_head_payload = cols < head_payload
    local_head = cols // per_head
    field = cols - local_head * per_head
    head = FIRST_HEAD + local_head
    vector_base = row * NUM_HEADS * HEAD_DIM + head * HEAD_DIM

    value = tl.zeros((BLOCK,), dtype=tl.float32)
    q_mask = mask & in_head_payload & (field < HEAD_DIM)
    value += tl.load(Q + vector_base + field, mask=q_mask, other=0.0).to(tl.float32)
    k_field = field - HEAD_DIM
    k_mask = mask & in_head_payload & (k_field >= 0) & (k_field < HEAD_DIM)
    value += tl.load(K + vector_base + k_field, mask=k_mask, other=0.0).to(tl.float32)
    v_field = field - 2 * HEAD_DIM
    v_mask = mask & in_head_payload & (v_field >= 0) & (v_field < HEAD_DIM)
    value += tl.load(V + vector_base + v_field, mask=v_mask, other=0.0).to(tl.float32)
    scalar_mask = mask & in_head_payload & (field == 3 * HEAD_DIM)
    value += tl.load(
        SCALAR + row * NUM_HEADS + head,
        mask=scalar_mask,
        other=0.0,
    ).to(tl.float32)
    if LINEAR:
        shared_col = cols - head_payload
        shared_mask = mask & ~in_head_payload
        value += tl.load(
            SHARED + row * SHARED_WIDTH + shared_col,
            mask=shared_mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(OUT + row * row_width + cols, value, mask=mask)


@dataclass
class UlyssesRuntime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str
    sequence_length: int = 0
    splits: tuple[int, ...] = ()
    local_start: int = 0
    local_end: int = 0
    heads_per_rank: int = 0
    num_heads: int = 0
    softmax_ranks: int = 0
    softmax_head_splits: tuple[int, ...] = ()
    linear_head_splits: tuple[int, ...] = ()
    profile_enabled: bool = False
    softmax_dispatch_group: object | None = None
    linear_dispatch_group: object | None = None
    profile_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = field(
        default_factory=dict
    )

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        dist.barrier()

    def profile_start(self):
        if not self.profile_enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def profile_end(self, name: str, start) -> None:
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.profile_events.setdefault(name, []).append((start, end))

    def reset_profile(self) -> None:
        self.profile_events.clear()

    def profile_milliseconds(self) -> dict[str, float]:
        torch.cuda.synchronize(self.device)
        return {
            name: sum(start.elapsed_time(end) for start, end in events)
            for name, events in self.profile_events.items()
        }

    @property
    def branch_parallel(self) -> bool:
        return self.softmax_ranks > 0

    @property
    def branch_kind(self) -> str:
        if not self.branch_parallel:
            raise RuntimeError("branch_kind is only defined for branch-parallel Ulysses")
        return "softmax" if self.rank < self.softmax_ranks else "linear"

    @property
    def branch_head_splits(self) -> tuple[int, ...]:
        if not self.branch_parallel:
            raise RuntimeError("branch_head_splits requires branch-parallel Ulysses")
        return self.softmax_head_splits + self.linear_head_splits

    @property
    def branch_heads(self) -> int:
        return self.branch_head_splits[self.rank]

    @property
    def branch_first_head(self) -> int:
        splits = (
            self.softmax_head_splits
            if self.branch_kind == "softmax"
            else self.linear_head_splits
        )
        branch_rank = self.rank if self.branch_kind == "softmax" else self.rank - self.softmax_ranks
        return sum(splits[:branch_rank])

    def enable_branch_parallel(self, softmax_ranks: int) -> None:
        if not 0 < softmax_ranks < self.world_size:
            raise ValueError(
                f"softmax_ranks must lie in [1, {self.world_size - 1}], got {softmax_ranks}"
            )
        if self.sequence_length:
            raise RuntimeError("enable branch parallelism before the first forward")
        self.softmax_ranks = softmax_ranks

    def configure(self, sequence_length: int, num_heads: int) -> None:
        if not self.branch_parallel and num_heads % self.world_size:
            raise ValueError(
                f"Ulysses needs num_heads divisible by world_size; got "
                f"{num_heads} heads on {self.world_size} ranks"
            )
        splits = sequence_splits(sequence_length, self.world_size)
        if any(size <= 0 for size in splits):
            raise ValueError(
                f"Ulysses requires at least one row per rank; S={sequence_length}, "
                f"P={self.world_size}"
            )
        if self.sequence_length and self.sequence_length != sequence_length:
            raise RuntimeError(
                "one Ulysses process supports one packed sequence geometry; "
                f"already configured for {self.sequence_length}, got {sequence_length}"
            )
        self.sequence_length = sequence_length
        self.splits = splits
        self.local_start = sum(splits[: self.rank])
        self.local_end = self.local_start + splits[self.rank]
        self.num_heads = num_heads
        if self.branch_parallel:
            linear_ranks = self.world_size - self.softmax_ranks
            self.softmax_head_splits = balanced_splits(num_heads, self.softmax_ranks)
            self.linear_head_splits = balanced_splits(num_heads, linear_ranks)
            self.heads_per_rank = self.branch_heads
        else:
            self.heads_per_rank = num_heads // self.world_size

    def dispatch_fields_to_branches_overlapped(
        self,
        softmax_q: torch.Tensor,
        softmax_k: torch.Tensor,
        value: torch.Tensor,
        softmax_gate: torch.Tensor,
        linear_q: torch.Tensor,
        linear_k: torch.Tensor,
        linear_beta: torch.Tensor,
        linear_shared: torch.Tensor,
    ):
        """Launch softmax and linear A2As independently and wait only for this rank's branch."""
        if self.softmax_dispatch_group is None or self.linear_dispatch_group is None:
            raise RuntimeError("overlapped dispatch communicators were not initialized")
        local_rows = self.splits[self.rank]
        head_dim = value.shape[-1]
        per_head = 3 * head_dim + 1
        shared_width = linear_shared.shape[-1]

        profile_event = self.profile_start()
        soft_input_splits = [0] * self.world_size
        soft_specs = []
        soft_first = 0
        for target, heads in enumerate(self.softmax_head_splits):
            row_width = heads * per_head
            size = local_rows * row_width
            soft_input_splits[target] = size
            soft_specs.append((soft_first, heads, row_width, size))
            soft_first += heads
        soft_send = value.new_empty(sum(soft_input_splits))
        offset = 0
        for first_head, heads, row_width, size in soft_specs:
            _pack_branch_target_kernel[
                (local_rows, triton.cdiv(row_width, 256))
            ](
                softmax_q,
                softmax_k,
                value,
                softmax_gate,
                linear_shared,
                soft_send[offset : offset + size],
                local_rows,
                NUM_HEADS=self.num_heads,
                HEAD_DIM=head_dim,
                FIRST_HEAD=first_head,
                LOCAL_HEADS=heads,
                SHARED_WIDTH=shared_width,
                LINEAR=False,
                BLOCK=256,
                num_warps=4,
            )
            offset += size
        soft_row_width = (
            self.branch_heads * per_head if self.branch_kind == "softmax" else 0
        )
        soft_output_splits = [rows * soft_row_width for rows in self.splits]
        soft_recv = value.new_empty(sum(soft_output_splits))
        soft_work = dist.all_to_all_single(
            soft_recv,
            soft_send,
            output_split_sizes=soft_output_splits,
            input_split_sizes=soft_input_splits,
            group=self.softmax_dispatch_group,
            async_op=True,
        )

        linear_input_splits = [0] * self.world_size
        linear_specs = []
        linear_first = 0
        for branch_rank, heads in enumerate(self.linear_head_splits):
            target = self.softmax_ranks + branch_rank
            row_width = heads * per_head + shared_width
            size = local_rows * row_width
            linear_input_splits[target] = size
            linear_specs.append((linear_first, heads, row_width, size))
            linear_first += heads
        linear_send = value.new_empty(sum(linear_input_splits))
        offset = 0
        for first_head, heads, row_width, size in linear_specs:
            _pack_branch_target_kernel[
                (local_rows, triton.cdiv(row_width, 256))
            ](
                linear_q,
                linear_k,
                value,
                linear_beta,
                linear_shared,
                linear_send[offset : offset + size],
                local_rows,
                NUM_HEADS=self.num_heads,
                HEAD_DIM=head_dim,
                FIRST_HEAD=first_head,
                LOCAL_HEADS=heads,
                SHARED_WIDTH=shared_width,
                LINEAR=True,
                BLOCK=256,
                num_warps=4,
            )
            offset += size
        self.profile_end("branch_pack", profile_event)
        linear_row_width = (
            self.branch_heads * per_head + shared_width
            if self.branch_kind == "linear"
            else 0
        )
        linear_output_splits = [rows * linear_row_width for rows in self.splits]
        linear_recv = value.new_empty(sum(linear_output_splits))
        linear_work = dist.all_to_all_single(
            linear_recv,
            linear_send,
            output_split_sizes=linear_output_splits,
            input_split_sizes=linear_input_splits,
            group=self.linear_dispatch_group,
            async_op=True,
        )

        profile_event = self.profile_start()
        if self.branch_kind == "softmax":
            soft_work.wait()
            recv = soft_recv.view(self.sequence_length, soft_row_width)
            packed = recv.view(
                self.sequence_length,
                self.branch_heads,
                per_head,
            )
            shared = None
            pending = (linear_work, linear_send, linear_recv)
        else:
            linear_work.wait()
            recv = linear_recv.view(self.sequence_length, linear_row_width)
            head_payload = self.branch_heads * per_head
            packed = recv[:, :head_payload].view(
                self.sequence_length,
                self.branch_heads,
                per_head,
            )
            shared = recv[:, head_payload:]
            pending = (soft_work, soft_send, soft_recv)
        self.profile_end("branch_relevant_wait", profile_event)
        return packed, shared, pending

    def branches_to_sequence(self, branch_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return disjoint branch/head shards to all eight sequence owners in one A2A.

        The first result is ``[S_rank, H, D]`` softmax output and the second has the
        same shape for the linear branch.  No projected hidden-state partials cross the
        fabric, which keeps this at the information-theoretic 2*S*H*D payload rather
        than an eight-rank hidden-width reduce-scatter.
        """
        if not self.branch_parallel:
            raise RuntimeError("branches_to_sequence requires branch-parallel Ulysses")
        if branch_tensor.ndim != 3 or branch_tensor.shape[:2] != (
            self.sequence_length,
            self.branch_heads,
        ):
            raise ValueError(
                f"branch output has {tuple(branch_tensor.shape)}, expected "
                f"({self.sequence_length}, {self.branch_heads}, D)"
            )
        width = branch_tensor.shape[-1]
        input_splits = [rows * self.branch_heads * width for rows in self.splits]
        local_rows = self.splits[self.rank]
        output_splits = [local_rows * heads * width for heads in self.branch_head_splits]

        recv = branch_tensor.new_empty(sum(output_splits))
        profile_event = self.profile_start()
        dist.all_to_all_single(
            recv,
            branch_tensor.contiguous().view(-1),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )
        self.profile_end("output_a2a", profile_event)

        profile_event = self.profile_start()
        chunks = []
        offset = 0
        for size, heads in zip(output_splits, self.branch_head_splits):
            chunks.append(recv[offset : offset + size].view(local_rows, heads, width))
            offset += size
        softmax = torch.cat(chunks[: self.softmax_ranks], dim=1)
        linear = torch.cat(chunks[self.softmax_ranks :], dim=1)
        self.profile_end("output_unpack", profile_event)
        return softmax, linear

    def sequence_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """[S_rank, H, D] -> [S, H/P, D] with one all-to-all."""
        local_rows, heads, width = tensor.shape
        if local_rows != self.splits[self.rank]:
            raise ValueError(f"local rows {local_rows} != configured {self.splits[self.rank]}")
        if heads != self.heads_per_rank * self.world_size:
            raise ValueError(f"heads {heads} != {self.heads_per_rank * self.world_size}")

        send = (
            tensor.view(local_rows, self.world_size, self.heads_per_rank, width)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(-1)
        )
        unit = self.heads_per_rank * width
        input_splits = [local_rows * unit] * self.world_size
        output_splits = [rows * unit for rows in self.splits]
        recv = tensor.new_empty(sum(output_splits))
        dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )
        return recv.view(self.sequence_length, self.heads_per_rank, width)

    def heads_to_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        """[S, H/P, D] -> [S_rank, H, D] with one all-to-all."""
        sequence_length, local_heads, width = tensor.shape
        if sequence_length != self.sequence_length or local_heads != self.heads_per_rank:
            raise ValueError(
                f"head shard has {tuple(tensor.shape[:2])}, expected "
                f"({self.sequence_length}, {self.heads_per_rank})"
            )
        unit = self.heads_per_rank * width
        input_splits = [rows * unit for rows in self.splits]
        local_rows = self.splits[self.rank]
        output_splits = [local_rows * unit] * self.world_size
        recv = tensor.new_empty(sum(output_splits))
        dist.all_to_all_single(
            recv,
            tensor.contiguous().view(-1),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )
        return (
            recv.view(self.world_size, local_rows, self.heads_per_rank, width)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(local_rows, self.heads_per_rank * self.world_size, width)
        )

    def gather_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather a sequence shard, with at most P-1 padded rows."""
        local_rows = tensor.shape[0]
        max_rows = max(self.splits)
        if local_rows < max_rows:
            padded = tensor.new_zeros((max_rows, *tensor.shape[1:]))
            padded[:local_rows].copy_(tensor)
        else:
            padded = tensor
        gathered = tensor.new_empty((self.world_size * max_rows, *tensor.shape[1:]))
        dist.all_gather_into_tensor(gathered, padded.contiguous())

        # The production split has every short/padded shard last, hence valid rows are
        # one prefix.  Retain a correct fallback for very short unit-test geometries.
        if self.splits[:-1] == (max_rows,) * (self.world_size - 1):
            return gathered[: self.sequence_length]
        chunks = gathered.view(self.world_size, max_rows, *tensor.shape[1:])
        return torch.cat([chunks[r, :n] for r, n in enumerate(self.splits)], dim=0)

    def _local_video_frame_sums(self, local_x: torch.Tensor, layout) -> torch.Tensor:
        sums = torch.zeros(
            layout.num_frames,
            local_x.shape[-1],
            device=local_x.device,
            dtype=torch.float32,
        )
        positions = torch.arange(self.local_start, self.local_end, device=local_x.device)
        is_video = (positions >= layout.video_start) & (positions < layout.video_end)
        if is_video.any():
            frames = torch.div(
                positions[is_video] - layout.video_start,
                layout.tokens_per_frame,
                rounding_mode="floor",
            )
            sums.index_add_(0, frames, local_x[is_video].float())
        return sums

    def video_frame_mean_async(self, local_x: torch.Tensor, layout):
        """Launch the frame-sum all-reduce so QKV, gates and dispatch can hide it."""
        sums = self._local_video_frame_sums(local_x, layout)
        work = dist.all_reduce(sums, async_op=True)
        return sums, work


def init_ulysses(backend: str = "nccl", *, profile_enabled: bool = False) -> UlyssesRuntime:
    """Initialise the torchrun process group and bind this process to LOCAL_RANK.

    ``profile_enabled`` (config ``parallel.profile``) records CUDA events around every
    forward section -- for profiling only, it is not free. The branch-parallel forward
    has one shape (low-rank gate dispatch, direct Triton packing, asynchronous frame
    mean, split A2As); the two dispatch communicators it needs are created here.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError(
            "infer_ulysses.py must be launched with torchrun and at least two ranks"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    actual_world = dist.get_world_size()
    if actual_world != world_size:
        raise RuntimeError(f"WORLD_SIZE={world_size}, process group has {actual_world}")
    runtime = UlyssesRuntime(
        rank=rank,
        world_size=actual_world,
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
        backend=backend,
        profile_enabled=profile_enabled,
    )

    ranks = list(range(actual_world))
    runtime.softmax_dispatch_group = dist.new_group(ranks=ranks, backend=backend)
    runtime.linear_dispatch_group = dist.new_group(ranks=ranks, backend=backend)
    return runtime
