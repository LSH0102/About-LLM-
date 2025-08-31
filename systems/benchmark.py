
import torch
import time
import numpy as np
from torch.nn.functional import scaled_dot_product_attention

from Attention_triton import Attention_Triton

def generate_inputs(batch_size, Rq, Rk, d, device):
    
    Q = torch.randn(batch_size, Rq, d, device=device, requires_grad=True)
    K = torch.randn(batch_size, Rk, d, device=device, requires_grad=True)
    V = torch.randn(batch_size, Rk, d, device=device, requires_grad=True)
    grad_output = torch.randn_like(Q)
    return Q, K, V, grad_output

def benchmark_forward_backward(attention_func, Q, K, V, grad_output, is_causal=False, warmup=5, iterations=30):
    for _ in range(warmup):
        output = attention_func(Q, K, V, is_causal=is_causal)
        output.backward(grad_output, retain_graph=True)
        Q.grad.zero_()
        K.grad.zero_()
        V.grad.zero_()
        if Q.device.type == "cuda":
            torch.cuda.synchronize()

   
    forward_times = []
    backward_times = []
    total_times = []
    for _ in range(iterations):
        
        start = time.perf_counter()
        output = attention_func(Q, K, V, is_causal=is_causal)
        if Q.device.type == "cuda":
            torch.cuda.synchronize()
        forward_end = time.perf_counter()

        
        output.backward(grad_output, retain_graph=True)
        if Q.device.type == "cuda":
            torch.cuda.synchronize()
        backward_end = time.perf_counter()

        forward_times.append(forward_end - start)
        backward_times.append(backward_end - forward_end)
        total_times.append(backward_end - start)
        Q.grad.zero_()
        K.grad.zero_()
        V.grad.zero_()

    forward_avg = np.mean(forward_times)
    backward_avg = np.mean(backward_times)
    total_avg = np.mean(total_times)
    return forward_avg, backward_avg, total_avg

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备：{device.type.upper()}\n")

    test_cases = [
        {"case": "Rq:128 Rk:128 d:64", "batch": 32, "Rq": 128, "Rk": 128, "d": 64},
        {"case": "Rq:256 Rk:256 d:64", "batch": 32, "Rq": 256, "Rk": 256, "d": 64},
        {"case": "Rq:512 Rk:512 d:64", "batch": 32, "Rq": 512, "Rk": 512, "d": 64},
    ]

    def standard_attention(Q, K, V, is_causal):
        Q_ = Q.unsqueeze(1)  
        K_ = K.unsqueeze(1)  
        V_ = V.unsqueeze(1)  
        output = scaled_dot_product_attention(Q_, K_, V_, is_causal=is_causal)
        return output.squeeze(1)  

    def triton_attention(Q, K, V, is_causal):
        return Attention_Triton.apply(Q, K, V, is_causal)

    print(f"{'测试用例':<20} | {'阶段':<12} | {'标准实现 (ms)':<18} | {'Triton实现 (ms)':<18} | {'加速比'}")
    print("-" * 80)

    for case in test_cases:
       
        batch, Rq, Rk, d = case["batch"], case["Rq"], case["Rk"], case["d"]
        Q, K, V, grad_output = generate_inputs(batch, Rq, Rk, d, device)
        is_causal = True

       
        std_fwd, std_bwd, std_total = benchmark_forward_backward(
            standard_attention, Q, K, V, grad_output, is_causal=is_causal
        )

       
        tri_fwd, tri_bwd, tri_total = benchmark_forward_backward(
            triton_attention, Q, K, V, grad_output, is_causal=is_causal
        )

        
        rows = [
            (case["case"], "前向传播", std_fwd, tri_fwd),
            (case["case"], "反向传播", std_bwd, tri_bwd),
            (case["case"], "总时间", std_total, tri_total),
        ]
        for row in rows:
            case_name, phase, std_time, tri_time = row
            speedup = std_time / tri_time if tri_time > 0 else 0
            print(
                f"{case_name:<20} | {phase:<12} | "
                f"{std_time*1000:.2f} ms       | {tri_time*1000:.2f} ms       | "
                f"{speedup:.2f}x"
            )
        print("-" * 80)

if __name__ == "__main__":
    main()
