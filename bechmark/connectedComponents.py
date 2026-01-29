import time
import cv2
import torch
import fastcv
import numpy as np

def generate_image(size, mode):
    if mode == "noise":
        return (np.random.rand(size, size) > 0.9).astype(np.uint8) * 255
    elif mode == "blobs":
        small_size = max(size // 64, 1)
        small_img = np.random.randint(0, 2, (small_size, small_size), dtype=np.uint8)
        img = cv2.resize(small_img, (size, size), interpolation=cv2.INTER_NEAREST)
        return img * 255

def verify_correctness():
    print("--- VERIFICATION ---")
    size = 1024
    for mode in ["noise", "blobs"]:
        img_np = generate_image(size, mode)
        # cv2.imwrite(f"{mode}.png", img_np)
        img_torch = torch.from_numpy(img_np).cuda()

        n_cv, _ = cv2.connectedComponents(img_np, connectivity=4)

        out_naive = fastcv.naiveConnectedComponents(img_torch)
        n_naive = len(torch.unique(out_naive))

        out_opt = fastcv.connectedComponents(img_torch)
        n_opt = len(torch.unique(out_opt))


        match = (n_cv == n_naive == n_opt)
        status = "SUCCESS" if match else "FAILURE"
        print(f"[{mode.upper()}] CV: {n_cv} | Naive: {n_naive} | Opt: {n_opt} |  -> {status}")
    print("")

def benchmark_ccl(sizes=[1024, 2048, 4096], runs=20):
    print("--- BENCHMARK RESULTS (ms) ---")
    
    modes = ["noise", "blobs"]
    
    for mode in modes:
        print(f"\n=== DATASET: {mode.upper()} ===")
        print(f"{'Size':<12} {'OpenCV':<12} {'Naive':<12} {'Opt':<12}")

        for size in sizes:
            img_np = generate_image(size, mode)
            img_torch = torch.from_numpy(img_np).cuda()

            start = time.perf_counter()
            for _ in range(runs):
                cv2.connectedComponents(img_np, connectivity=4)
            cv_time = (time.perf_counter() - start) / runs * 1000

            fastcv.naiveConnectedComponents(img_torch)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(runs):
                fastcv.naiveConnectedComponents(img_torch)
            torch.cuda.synchronize()
            naive_time = (time.perf_counter() - start) / runs * 1000

            fastcv.connectedComponents(img_torch)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(runs):
                fastcv.connectedComponents(img_torch)
            torch.cuda.synchronize()
            opt_time = (time.perf_counter() - start) / runs * 1000


            print(f"{size}x{size:<7} {cv_time:<12.2f} {naive_time:<12.2f} {opt_time:<12.2f}")

if __name__ == "__main__":
    verify_correctness()
    benchmark_ccl()