# step1. generate parameter combinations
#1) BS
def generate_BS_params(n_sets, seed=None):
    if seed is not None:
        np.random.seed(seed)

    S0 = np.random.normal(loc=1.0, scale=0.2, size=n_sets)  # S0 ~ N(1, 0.2^2) Greek 계산시 delta를 구할 경우 1로 안둠.
    r = np.random.uniform(0, 0.1, n_sets)      # r ~ U(0, 0.1)
    sigma = np.random.uniform(0.001, 1, n_sets)# σ ~ U(0.001, 1)
    T = np.random.uniform(0.1, 3, n_sets)      # T ~ U(0.1, 3)
    K = 1                                      # K = 1

    params = np.stack([S0, K, r, sigma, T], axis=1)
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

def filter_milestein(params): # milestein condition
    _, lamb, v_bar, epsilon, *_ = params.T
    mask = 4 * lamb * v_bar > epsilon**2
    return params[mask]

def generate_valid_params(n_sets, seed=None):
    raw = generate_heston_params(int(n_sets * 1.1), seed)
    filtered = filter_milestein(raw)
    
    while len(filtered) < n_sets:
        extra = generate_heston_params(n_sets)
        extra = filter_milestein(extra)
        filtered = np.vstack([filtered, extra])
    
    return filtered[:n_sets]

def min_max_normalize(data):
    return (data - data.min()) / (data.max() - data.min())







#step1-1
BS_paras = generate_BS_params(n_sets=100*(2**16), seed=1234)
BS_paras_scaled = BS_paras.copy()

for col in range(5):
    if col == 0:
        BS_paras_scaled[:, col] = (BS_paras[:, col] - 1.0) / 0.2
    else:
        BS_paras_scaled[:, col] = min_max_normalize(BS_paras[:, col])





#step1-2 
Hes_paras = generate_valid_params(n_sets=100*(2**16), seed=1234) # 4.1초

Hes_paras_scaled = Hes_paras.copy()
for col in range(7):
    Hes_paras_scaled[:, col] = min_max_normalize(Hes_paras[:, col])

ETA_PATH = "/mnt/d/heston_eta.h5" # 10초
with h5py.File(ETA_PATH, "w") as f:
    f.create_dataset("etas", data=Hes_paras,
                     maxshape=(None, 7), chunks=(10240, 7), compression="gzip")

"""
print(f"\n파라미터 범위 확인:")
names = ['r', 'λ', 'v_bar', 'ξ', 'ρ', 'Y₀', 'T']
for i, name in enumerate(names):
    print(f"{name}: min={params[:,i].min():.4f}, max={params[:,i].max():.4f}, mean={params[:,i].mean():.4f}")
"""


# step2. generate path
# 1) Heston
# MC.py -> generate_heston_paths

#step2-2
ETA_PATH  = "/mnt/d/heston_eta.h5"
SAVE_PATH = "/mnt/d/heston_dataset.h5"


for i in range(9,10):
    print(i)
    with h5py.File(ETA_PATH, "r") as ef:
        Hes_paras = ef["etas"][:]  # (2**16)*100개, 여기서 target*num개 만큼씩 가져오게 해도 됨.

    target = (2**16) * 10 # 3.6h 걸림
    true = 0
    fail = 0 # 1116 + 1128 + 1114 + 1111 + 1084 + 1127 + 1143 + 1058 + 1089 + 1054
    num = i # 9까지, done : 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 
    start = time.time()

    # CPU 병렬, 멀티 쓰레딩은 성능 안좋음
    N_WORKERS = 14 # 1만개 기준 = 14:207s/28:192s, 이용률 2배차이
    BATCH_SIZE = N_WORKERS * 10


    def _worker(args):
        eta, n_paths, dt = args
        return generate_heston_paths(eta, n_paths=n_paths, dt=dt)

    with h5py.File(SAVE_PATH, "a") as f, ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        if "paths" not in f:
            f.create_dataset("paths", shape=(0, 4), maxshape=(None, 4),
                            dtype="float64", chunks=(10240, 4), compression="gzip")

        for i in range(0, target, BATCH_SIZE):
            eta_batch = Hes_paras[i + target*num : i + BATCH_SIZE + target*num]
                    
            futures = {executor.submit(_worker, (eta, 2**10, 0.001)): (i + target*num + k, eta)
                    for k, eta in enumerate(eta_batch)}

            for future in as_completed(futures):
                ori_idx, eta = futures[future]
                XT, YT, MT, mask = future.result()

                if mask:
                    rows = np.column_stack([
                        np.full(len(XT), ori_idx),
                        XT, YT, MT
                    ])
                    f["paths"].resize(f["paths"].shape[0] + len(rows), axis=0)
                    f["paths"][-len(rows):] = rows

                    true += 1
                    if true % 20000 == 0:
                        print(f"[{true}/{target}] fail: {fail}")
                else:
                    fail += 1

                if target <= true:
                    break
            else:
                continue
            break
        else:
            Hes_paras = generate_valid_params(n_sets=(target - true), seed=None)

    elapsed = time.time() - start
    print(f"done. true: {true}, fail: {fail}")
    print(f"elapsed: {elapsed:.1f}s ({elapsed/3600:.2f}h)")


