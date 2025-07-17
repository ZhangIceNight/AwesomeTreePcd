import torch
import time
from ptflops import get_model_complexity_info
from models import get_model
from tqdm import *
# 输入参数
B, N, C = 1, 1024*1, 3  # B: batch_size, N: 点数, C: 每点维度
input_shape = (C, N)

model_name = 'pcm'
# # 模型加载
model = get_model(model_name, cls_dim=2).cuda().eval()




# ----------------------------------------
# 1. 参数量 和 FLOPs
def get_flops_params():
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, input_shape,
                                                  as_strings=True,
                                                  print_per_layer_stat=False,
                                                  verbose=False)
    print(f"[Params] 参数量: {params}")
    print(f"[FLOPs] 浮点运算: {flops}")

# ----------------------------------------
# 2. 推理速度（FPS）
def get_fps():
    dummy_input = torch.randn(B, C, N).cuda()
    for _ in range(10):  # warmup
        _ = model(dummy_input)
    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in tqdm(range(20), desc="[FPS TEST]"):
            _ = model(dummy_input)
        torch.cuda.synchronize()
    end = time.time()
    fps = 20 / (end - start)
    print(f"[FPS] 推理速度: {fps:.2f} frame/s")

# ----------------------------------------
# 3. GPU 显存占用
def get_memory():
    dummy_input = torch.randn(B, C, N).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_input)
    mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[GPU Memory] 显存峰值: {mem:.2f} MB")

# ----------------------------------------
if __name__ == "__main__":
    print(f"✅ {model_name} 性能分析中...")
    get_flops_params()
    get_fps()
    # get_memory()
