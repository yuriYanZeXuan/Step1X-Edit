import torch

def demonstrate_t_vec():
    """演示t_vec代码的作用"""
    
    # 模拟图像批次数据（使用CPU）
    batch_size = 4
    img = torch.randn(batch_size, 3, 512, 512, dtype=torch.float32, device='cpu')
    t_curr = 0.8  # 当前时间步
    
    print(f"原始图像张量形状: {img.shape}")
    print(f"图像数据类型: {img.dtype}")
    print(f"图像设备: {img.device}")
    print(f"当前时间步: {t_curr}")
    
    # 创建时间步向量（这就是原代码的作用）
    t_vec = torch.full(
        (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
    )
    
    print(f"\n创建的时间步向量: {t_vec}")
    print(f"时间步向量形状: {t_vec.shape}")
    print(f"时间步向量数据类型: {t_vec.dtype}")
    print(f"时间步向量设备: {t_vec.device}")
    
    # 验证每个批次样本都有相同的时间步
    print(f"\n验证所有批次样本的时间步是否相同: {torch.all(t_vec == t_curr)}")
    
    # 模拟在扩散模型中的使用
    print(f"\n在扩散模型中的典型使用:")
    print(f"- 时间步向量用于控制去噪强度")
    print(f"- 每个批次样本使用相同的时间步进行并行处理")
    print(f"- 时间步从1.0逐渐减少到0.0，控制噪声去除程度")

def demonstrate_with_different_scenarios():
    """演示不同场景下的t_vec创建"""
    
    print("\n" + "="*50)
    print("不同场景示例:")
    
    # 场景1: 单张图像
    img_single = torch.randn(1, 3, 256, 256, dtype=torch.float16, device='cpu')
    t_curr = 0.5
    t_vec_single = torch.full((img_single.shape[0],), t_curr, dtype=img_single.dtype, device=img_single.device)
    print(f"单张图像场景: {t_vec_single}")
    
    # 场景2: 大批次图像
    img_batch = torch.randn(8, 3, 512, 512, dtype=torch.float32, device='cpu')
    t_curr = 0.2
    t_vec_batch = torch.full((img_batch.shape[0],), t_curr, dtype=img_batch.dtype, device=img_batch.device)
    print(f"大批次场景: {t_vec_batch}")
    
    # 场景3: 不同数据类型
    img_int = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.int32, device='cpu')
    t_curr = 0.9
    t_vec_int = torch.full((img_int.shape[0],), t_curr, dtype=img_int.dtype, device=img_int.device)
    print(f"整数类型场景: {t_vec_int}")

if __name__ == "__main__":
    demonstrate_t_vec()
    demonstrate_with_different_scenarios() 