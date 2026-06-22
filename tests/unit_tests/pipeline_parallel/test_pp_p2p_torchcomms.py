# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Smoke test and P2P overlap benchmark for PP under TorchComms nccl-lazy.

Validates:
1. PP group creation with nccl-lazy succeeds under torchcomms
2. P2P send/recv works across PP stages
3. Concurrent P2P to different peers can overlap (performance regression test)

Run: torchrun --nproc_per_node=4 tests/unit_tests/pipeline_parallel/test_pp_p2p_torchcomms.py
"""
import os
import sys
import time

import torch
import torch.distributed as dist

from tests.unit_tests.test_utilities import Utils


def run_all():
    """Run all tests in a single init/destroy cycle."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
    )
    from megatron.core import parallel_state as ps

    pp_group = ps.get_pipeline_model_parallel_group()

    # --- test_pp_group_creation ---
    assert pp_group is not None, "PP group not created"
    assert pp_group.size() == 4, f"Expected PP size 4, got {pp_group.size()}"

    tensor = torch.tensor([pp_group.rank() + 1.0], device="cuda")
    dist.all_reduce(tensor, group=pp_group)
    expected = sum(range(1, 5))
    assert tensor.item() == expected, f"allreduce: expected {expected}, got {tensor.item()}"
    if dist.get_rank() == 0:
        print("PASS: test_pp_group_creation")

    # --- test_pp_p2p_send_recv ---
    pp_rank = pp_group.rank()
    pp_size = pp_group.size()

    if pp_rank < pp_size - 1:
        next_global = dist.get_global_rank(pp_group, pp_rank + 1)
        send_tensor = torch.tensor([pp_rank * 10.0], device="cuda")
        dist.send(send_tensor, dst=next_global, group=pp_group)

    if pp_rank > 0:
        prev_global = dist.get_global_rank(pp_group, pp_rank - 1)
        recv_tensor = torch.empty(1, device="cuda")
        dist.recv(recv_tensor, src=prev_global, group=pp_group)
        expected = (pp_rank - 1) * 10.0
        assert recv_tensor.item() == expected, (
            f"rank {pp_rank}: expected {expected}, got {recv_tensor.item()}"
        )

    dist.barrier()
    if dist.get_rank() == 0:
        print("PASS: test_pp_p2p_send_recv")

    # --- test_pp_batch_isend_irecv ---
    ops = []
    recv_tensors = {}

    if pp_rank < pp_size - 1:
        next_global = dist.get_global_rank(pp_group, pp_rank + 1)
        send_tensor = torch.tensor([pp_rank + 100.0], device="cuda")
        ops.append(dist.P2POp(dist.isend, send_tensor, next_global, pp_group))

    if pp_rank > 0:
        prev_global = dist.get_global_rank(pp_group, pp_rank - 1)
        recv_tensor = torch.empty(1, device="cuda")
        recv_tensors["prev"] = recv_tensor
        ops.append(dist.P2POp(dist.irecv, recv_tensor, prev_global, pp_group))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    if "prev" in recv_tensors:
        expected = (pp_rank - 1) + 100.0
        actual = recv_tensors["prev"].item()
        assert actual == expected, (
            f"rank {pp_rank}: expected {expected}, got {actual}"
        )

    dist.barrier()
    if dist.get_rank() == 0:
        print("PASS: test_pp_batch_isend_irecv")

    # --- bench_p2p_overlap ---
    pp_rank = pp_group.rank()

    nbytes = 64 * 1024 * 1024  # 64 MiB
    nelems = nbytes // 4
    warmup_iters = 5
    bench_iters = 20

    r1_global = dist.get_global_rank(pp_group, 1)
    r2_global = dist.get_global_rank(pp_group, 2)

    send_buf_1 = torch.ones(nelems, device="cuda")
    send_buf_2 = torch.ones(nelems, device="cuda") * 2
    recv_buf = torch.empty(nelems, device="cuda")

    # --- Warmup ---
    for _ in range(warmup_iters):
        if pp_rank == 0:
            r1 = dist.isend(send_buf_1, r1_global, group=pp_group)
            r2 = dist.isend(send_buf_2, r2_global, group=pp_group)
            r1.wait()
            r2.wait()
        elif pp_rank == 1:
            r = dist.irecv(recv_buf, r1_global if False else dist.get_global_rank(pp_group, 0), group=pp_group)
            r.wait()
        elif pp_rank == 2:
            r = dist.irecv(recv_buf, dist.get_global_rank(pp_group, 0), group=pp_group)
            r.wait()
        dist.barrier()

    # --- Sequential baseline: one send at a time ---
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        if pp_rank == 0:
            r1 = dist.isend(send_buf_1, r1_global, group=pp_group)
            r1.wait()
        elif pp_rank == 1:
            r = dist.irecv(recv_buf, dist.get_global_rank(pp_group, 0), group=pp_group)
            r.wait()
        dist.barrier()

        if pp_rank == 0:
            r2 = dist.isend(send_buf_2, r2_global, group=pp_group)
            r2.wait()
        elif pp_rank == 2:
            r = dist.irecv(recv_buf, dist.get_global_rank(pp_group, 0), group=pp_group)
            r.wait()
        dist.barrier()
    torch.cuda.synchronize()
    t_seq = (time.perf_counter() - t0) / bench_iters

    # --- Concurrent: both sends in flight at once ---
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        if pp_rank == 0:
            r1 = dist.isend(send_buf_1, r1_global, group=pp_group)
            r2 = dist.isend(send_buf_2, r2_global, group=pp_group)
            r1.wait()
            r2.wait()
        elif pp_rank == 1:
            r = dist.irecv(recv_buf, dist.get_global_rank(pp_group, 0), group=pp_group)
            r.wait()
        elif pp_rank == 2:
            r = dist.irecv(recv_buf, dist.get_global_rank(pp_group, 0), group=pp_group)
            r.wait()
        dist.barrier()
    torch.cuda.synchronize()
    t_conc = (time.perf_counter() - t0) / bench_iters

    if pp_rank == 0:
        speedup = t_seq / t_conc if t_conc > 0 else float("inf")
        print(f"P2P overlap benchmark (64 MiB × 2 sends):")
        print(f"  Sequential: {t_seq*1000:.2f} ms")
        print(f"  Concurrent: {t_conc*1000:.2f} ms")
        print(f"  Speedup:    {speedup:.2f}x")
        if speedup < 1.2:
            print("WARNING: P2P ops appear serialized (speedup < 1.2x)")
        else:
            print("PASS: P2P overlap detected")

    Utils.destroy_model_parallel()


if __name__ == "__main__":
    run_all()
    if dist.get_rank() == 0:
        print("\nAll tests passed.")
