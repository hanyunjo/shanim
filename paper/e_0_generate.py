from __future__ import annotations

import numpy as np
import os
import glob
import time
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import math
import re



# step1. generate parameter combinations
#1) BS
def generate_BS_params(n_sets, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # S0 = np.random.normal(loc=1.0, scale=0.2, size=n_sets)  # S0 ~ N(1, 0.2^2) Greek 계산시 delta를 구할 경우 1로 안둠.
    # K = np.ones(n_sets)                        # K = 1
    r = np.random.uniform(0, 0.1, n_sets)      # r ~ U(0, 0.1)
    sigma = np.random.uniform(0.001, 1, n_sets)# σ ~ U(0.001, 1)
    T = np.random.uniform(0.1, 3, n_sets)      # T ~ U(0.1, 3)

    params = np.stack([r, sigma, T], axis=1)
    return params

def generate_BS_params_clip(n_sets, seed=None):
    rng = np.random.default_rng(seed)

    r = rng.integers(0, 1001, size=n_sets) / 10000.0
    sigma = rng.integers(10, 10001, size=n_sets) / 10000.0
    T = rng.integers(1000, 30001, size=n_sets) / 10000.0

    params = np.stack([r, sigma, T], axis=1)
    return params

#2) heston
def generate_heston_params(n_sets, seed=None):
    if seed is not None:
        np.random.seed(seed)
    r = np.random.uniform(0, 0.1, n_sets)      # r ~ U(0, 0.1)
    lamb = np.random.beta(2, 18, n_sets) * 20  # λ ~ Beta(2, 18) × 20
    v_bar = np.random.beta(1, 19, n_sets)      # v_bar ~ Beta(1, 19)
    epsilon = np.random.uniform(0.1, 1, n_sets)# ξ ~ U(0.1, 1)
    rho = np.random.uniform(-1, 0, n_sets)     # ρ ~ U(-1, 0)
    Y0 = np.random.beta(1, 19, n_sets)         # Y₀ ~ Beta(1, 19)
    T = np.random.uniform(0.1, 3, n_sets)      # T ~ U(0.1, 3)
        
    params = np.stack([r, lamb, v_bar, epsilon, rho, Y0, T], axis=1)
    return params

def generate_heston_params_clip(n_sets, seed=None):
    rng = np.random.default_rng(seed)

    r = rng.integers(0, 1001, size=n_sets) / 10000.0            # r ~ U(0, 0.1)
    epsilon = rng.integers(1000, 10001, size=n_sets) / 10000.0  # ξ, xi ~ U(0.1, 1)   
    rho = rng.integers(-10000, 1, size=n_sets) / 10000.0        # ρ ~ U(-1, 0)
    T = rng.integers(1000, 30001, size=n_sets) / 10000.0        # T ~ U(0.1, 3)
    lamb = np.clip(
        np.round(rng.beta(2,18,n_sets)*20, 4),                  # λ, kappa ~ Beta(2, 18) × 20
        0.0001, 19.9999
    )
    v_bar = np.clip(
        np.round(rng.beta(1,19, n_sets), 4),                    # v_bar, theta ~ Beta(1, 19)
        0.0001, 0.9999
    )
    Y0 = np.clip(
        np.round(rng.beta(1,19, n_sets), 4),                    # Y₀ ~ Beta(1, 19)
        0.0001, 0.9999
    )
        
    params = np.stack([r, lamb, v_bar, epsilon, rho, Y0, T], axis=1)
    return params

def filter_milestein(params): # milestein condition
    _, lamb, v_bar, epsilon, *_ = params.T
    mask = 4 * lamb * v_bar > epsilon**2
    return params[mask]

def generate_hes_valid_params(n_sets, seed=None):
    raw = generate_heston_params(int(n_sets * 1.1), seed)
    filtered = filter_milestein(raw)
    
    while len(filtered) < n_sets:
        extra = generate_heston_params(n_sets)
        extra = filter_milestein(extra)
        filtered = np.vstack([filtered, extra])
    
    return filtered[:n_sets]

def min_max_normalize(data):
    return (data - data.min()) / (data.max() - data.min())








# step2. generate path
# 1) Heston
#step2-2
def gen_bs_paths(eta, S0=1.0, B=0.0, n_paths=2**10, dt=0.001 ):
    r, sigma, T = eta
    n_steps = int(T / dt)

    W = np.random.randn(n_paths, n_steps)
    X = np.zeros((n_paths, n_steps + 1))  # X_t = log(S_t/S0), X_0=0

    knocked = np.zeros(n_paths, dtype=bool)
    M_T     = np.zeros(n_paths)
    if B != 0:
        log_S0B = np.log(S0 / B)

    for i in range(n_steps):
        X[:, i+1] = X[:, i] + (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*W[:, i]

        # reflection principel correction
        if B != 0:
            a = log_S0B + X[:, i] 
            b = log_S0B + X[:, i+1] 

            valid   = (a > 0) & (b > 0)  # 두 점 모두 barrier 위
            p_cross = np.where(valid, np.exp(-2 * a * b / (sigma**2 * dt + 1e-10)), 0.0) # 베리어 위에 두 점이 있을 때 베리어를 건드릴 확률
            u = np.random.uniform(size=n_paths) # 0~1사이의 랜덤 변수, 베르누이 시행
            new_knocked = valid & (u < p_cross)

            reflected = -2 * log_S0B - X[:, i+1] # 반사된 점
            M_T = np.where(new_knocked & ~knocked, reflected, M_T)

            knocked = knocked | new_knocked

    X_T = X[:, -1]
    if B != 0:
        knocked = knocked | (X.min(axis=1) <= np.log(B/S0))
        M_T     = np.minimum(M_T, X.min(axis=1))
    else:
        M_T = X.min(axis=1)
    mask = filter_paths(X_T, T)

    return X_T, M_T, mask

def gen_heston_paths(eta, S0=1.0, B=0, n_paths=2**10, dt=0.001):
    r, kappa, theta, xi, rho, Y0, T = eta
    n_steps = int(T / dt)

    W1 = np.random.randn(n_paths, n_steps)
    W2 = np.random.randn(n_paths, n_steps)
    dWx = np.sqrt(dt) * W1
    dWy = np.sqrt(dt) * (rho * W1 + np.sqrt(1 - rho**2) * W2)

    X = np.zeros((n_paths, n_steps + 1))
    Y = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = 0.0
    Y[:, 0] = Y0

    knocked = np.zeros(n_paths, dtype=bool) # 1 : barrier를 건든 것
    M_T     = np.zeros(n_paths)
    
    if B != 0:
        log_S0B = np.log(S0 / B) 

    for i in range(n_steps):
        Y_t = np.maximum(Y[:, i], 0)
        # X = ln(S_t/S0)
        X[:, i+1] = X[:, i] + (r - 0.5 * Y_t) * dt + np.sqrt(Y_t) * dWx[:, i]
        Y[:, i+1] = (Y_t
                     + kappa * (theta - Y_t) * dt
                     + xi * np.sqrt(Y_t) * dWy[:, i]
                     + 0.25 * xi**2 * (dWy[:, i]**2 - dt))
        
        # reflection principel correction
        if B != 0:
            a = log_S0B + X[:, i] 
            b = log_S0B + X[:, i+1]    
            v_local = 0.5 * (Y_t + np.maximum(Y[:, i+1], 0))

            valid   = (a > 0) & (b > 0)
            p_cross = np.where(valid, np.exp(-2 * a * b / (v_local * dt + 1e-10)), 0.0)
            u = np.random.uniform(size=n_paths)
            new_knocked = valid & (u < p_cross)

            reflected = -2 * log_S0B - X[:, i+1]
            M_T = np.where(new_knocked & ~knocked, reflected, M_T)

            knocked = knocked | new_knocked

    XT = X[:, -1]
    if B != 0:
        knocked = knocked | (X.min(axis=1) <= np.log(B/S0))
        M_T     = np.minimum(M_T, X.min(axis=1))
    else:
        M_T = X.min(axis=1)
    mask = filter_paths(XT, T)

    return XT, M_T, mask

def filter_paths(XT, T):
    annual_return = XT / T
    check_mean = np.abs(annual_return.mean()) <= 0.3
    check_var = annual_return.var() <= 1.0
    return bool(check_mean and check_var)



def _flush_buffer(buffer, chunk_dir, prefix, chunk_idx):
    chunk     = np.concatenate(buffer, axis=0)
    save_path = os.path.join(chunk_dir, f"{prefix}_chunk_{chunk_idx:03d}.h5")
 
    while os.path.exists(save_path):
        chunk_idx += 1
        save_path = os.path.join(chunk_dir, f"{prefix}_chunk_{chunk_idx:03d}.h5")

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('paths', data=chunk,
                          chunks=(10240, 3), compression='gzip')

    print(f"  청크 저장: {save_path} ({len(chunk):,}행)")
    return chunk_idx + 1, [] 


def _bs_worker(args):
    eta, S0, B, n_paths, dt = args
    return gen_bs_paths(eta, S0, B, n_paths, dt) # XT, MT, mask

def _heston_worker(args):
    eta, S0, B, n_paths, dt = args
    return gen_heston_paths(eta, S0, B, n_paths, dt) # XT, MT, mask

def generate_dataset(eta_path, chunk_dir, model_type='hes', S0=1.0, B=0, 
                     chunk_size=2**16, n_workers=14):
    
    # target=100*(2**16)
    target_per_round = (2**16) * 10 # 3.6h 걸림, eta 개수, 1eta : 2**10 paths
    BATCH_SIZE       = n_workers * 10
    prefix           = 'heston' if model_type == 'hes' else 'bs'
    
    buffer     = []   # path 저장 버퍼
    buf_size   = 0    # 현재 버퍼 행 수
    chunk_idx  = 0    # 청크 파일 번호

    os.makedirs(chunk_dir, exist_ok=True)

    _worker = _heston_worker if model_type == 'hes' else _bs_worker

    with h5py.File(eta_path, "r") as f1:
        paras = f1["etas"][:] # (2**16)*100 x 7 or 3

    for i in range(0,10):
        print(f'round:{i}')
        true  = 0
        fail  = 0 # bs : 1,743,655 / hes: 11,111
        buffer   = []
        buf_size = 0 
        start = time.time()
        eta_offset = i * target_per_round

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for batch_start in range(0, target_per_round, BATCH_SIZE):
                eta_batch = paras[eta_offset + batch_start : eta_offset + batch_start + BATCH_SIZE]
                        
                futures = {
                    executor.submit(_worker, (eta, S0, B, 2**10, 0.001)): 
                    (eta_offset + batch_start + k, eta) for k, eta in enumerate(eta_batch)
                    }

                for future in as_completed(futures):
                    ori_idx, eta = futures[future]
                    XT, MT, mask = future.result()

                    if mask:
                        rows = np.column_stack([
                            np.full(len(XT), ori_idx), XT, MT
                        ])
                        buffer.append(rows)
                        buf_size += len(rows)
                        true += 1

                        if true % (chunk_size) == 0:
                            chunk_idx, buffer = _flush_buffer(
                                buffer, chunk_dir, prefix, chunk_idx)
                            buf_size = 0

                        if true % 20000 == 0:
                            print(f"[{true}/{target_per_round}] fail: {fail}")
                    else:
                        fail += 1

                    if target_per_round <= true:
                        break
                else:
                    continue
                break
            
            while true < target_per_round:
                shortage     = target_per_round - true
                extra_paras = generate_BS_params(n_sets=int(shortage * 1.3)) if model_type == 'bs' \
                    else generate_hes_valid_params(n_sets=int(shortage * 1))
                
                with h5py.File(eta_path, "a") as f2:
                    old_size = f2["etas"].shape[0] # append 전 크기
                    f2["etas"].resize(old_size + len(extra_paras), axis=0)
                    f2["etas"][-len(extra_paras):] = extra_paras

                for batch_start in range(0, len(extra_paras), BATCH_SIZE):
                    eta_batch = extra_paras[batch_start : batch_start + BATCH_SIZE]

                    if true >= target_per_round:
                        break

                    extra_futures = {
                        executor.submit(_worker, (eta, S0, B, 2**10, 0.001)):
                        (old_size + batch_start + j, eta) for j, eta in enumerate(eta_batch)
                    }

                    for future in as_completed(extra_futures):
                        ori_idx, eta = extra_futures[future]
                        XT, MT, mask = future.result()

                        if mask:
                            rows = np.column_stack([
                                np.full(len(XT), ori_idx), XT, MT
                            ])
                            buffer.append(rows)
                            buf_size += len(rows)
                            true += 1

                            if true % (chunk_size) == 0 and true > 0:
                                chunk_idx, buffer = _flush_buffer(
                                    buffer, chunk_dir, prefix, chunk_idx)
                                buf_size = 0
                                print(f"{time.time()-start}s")

                            if true % 20000 == 0:
                                print(f"[{true}/{target_per_round}] fail: {fail}")
                        else:
                            fail += 1
                        
                        if target_per_round <= true:
                            break

        elapsed = time.time() - start
        print(f"done. true: {true}, fail: {fail}")
        print(f"elapsed: {elapsed:.1f}s ({elapsed/3600:.2f}h)")



def compute_xm_stats(
    model_type="bs",
    chunk_dir=None,
    pattern="*.h5",
    expected_chunks=100,
    row_batch_size=1_000_000,
):
    if chunk_dir is None:
        if model_type == "bs":
            chunk_dir = Path("/mnt/d/bs_chunks_correction")
        elif model_type == "hes":
            chunk_dir = Path("/mnt/d/heston_chunks_correction")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    chunk_dir = Path(chunk_dir)
    chunk_paths = list_chunk_paths(chunk_dir, pattern, expected_chunks)

    n_total = 0
    x_sum = 0.0
    x_sum_sq = 0.0
    m_sum = 0.0
    m_sum_sq = 0.0

    for chunk_idx, chunk_path in enumerate(chunk_paths, start=1):
        with h5py.File(chunk_path, "r") as h5:
            data_ds = h5["paths"]

            if data_ds.ndim != 2 or data_ds.shape[1] < 3:
                raise ValueError(
                    f"Expected paths shape (N, >=3) with [eta_index, X_T, M_T], got {data_ds.shape}"
                )

            n_rows = data_ds.shape[0]

            for start in range(0, n_rows, row_batch_size):
                end = min(start + row_batch_size, n_rows)
                batch = np.asarray(data_ds[start:end])

                x_t = batch[:, 1].astype(np.float64, copy=False)
                m_t = batch[:, 2].astype(np.float64, copy=False)

                n_batch = end - start

                x_sum += float(x_t.sum())
                x_sum_sq += float((x_t * x_t).sum())

                m_sum += float(m_t.sum())
                m_sum_sq += float((m_t * m_t).sum())

                n_total += n_batch

        print(f"[{chunk_idx:03d}/{len(chunk_paths):03d}] done: {chunk_path.name} rows={n_rows:,}")

    if n_total == 0:
        raise ValueError("No rows were accumulated.")

    x_mean = x_sum / n_total
    m_mean = m_sum / n_total

    x_var = x_sum_sq / n_total - x_mean**2
    m_var = m_sum_sq / n_total - m_mean**2

    x_std = np.sqrt(max(x_var, 0.0))
    m_std = np.sqrt(max(m_var, 0.0))

    stats = {
        "model_type": model_type,
        "chunk_dir": str(chunk_dir),
        "num_chunks": len(chunk_paths),
        "n_total": n_total,
        "x_mean": x_mean,
        "x_std": x_std,
        "m_mean": m_mean,
        "m_std": m_std,
    }

    print("\n" + "=" * 64)
    print(f"{model_type} integrated X_T, M_T stats")
    print("=" * 64)
    print(f"chunks  : {len(chunk_paths)}")
    print(f"n_total : {n_total:,}")
    print(f"X_T mean: {x_mean:.10f}")
    print(f"X_T std : {x_std:.10f}")
    print(f"M_T mean: {m_mean:.10f}")
    print(f"M_T std : {m_std:.10f}")
    print("=" * 64)

    return stats