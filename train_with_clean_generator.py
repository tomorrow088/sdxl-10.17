"""
使用精简生成器的训练脚本
"""
import torch
import sys
import os
import traceback

# 让Python能找到项目中的其他模块（如config、sdxl_generator_clean等）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   

# 离线与稳定性环境（可根据需要注释）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 显存管理优化
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  

from config import model_config, training_config
from sdxl_generator_clean import SDXLCamouflageGenerator
from adversarial_trainer import AdversarialTrainer


def main():
    print("=== 使用精简SDXL生成器开始训练 ===", flush=True)   #flush=True：立即刷新输出缓冲区，确保信息立即显示

    try:
        # 全局性能优化
        import torch
        torch.set_float32_matmul_precision('high')
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    except Exception:
        pass

    try:
        # 设备检查
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}", flush=True)

        # 创建精简生成器
        print("\n🔄 初始化精简SDXL生成器...", flush=True)
        generator = SDXLCamouflageGenerator(
            model_path=model_config.sdxl_model_path,
            device=device,
            use_lora=True,
            use_smart_adapter=True
        )
        # 可选编译（PyTorch 2.0+），小心首次开销
        try:
            from config import training_config
            if getattr(training_config, 'use_torch_compile', False):
                generator_raw = generator
                generator_compiled = torch.compile(generator_raw, mode='reduce-overhead')
                generator = generator_compiled
                print('✅ 训练阶段使用 torch.compile（验证阶段将使用未编译模型）')
            else:
                generator_raw = generator
        except Exception as _e:
            print(f"⚠️ torch.compile 不可用: {_e}")
            generator_raw = generator

        from config import training_config as _tc
        print(f"训练步数: {_tc.train_num_steps}, 验证步数: {_tc.eval_num_steps}")

        # 创建对抗训练器（数据加载器由训练器内部创建）
        print("\n🔄 初始化对抗训练器...", flush=True)
        trainer = AdversarialTrainer(
            generator=generator,
            device=str(device)
        )
        # 验证阶段关闭compile：使用原始未编译生成器
        try:
            trainer.generator_for_validation = generator_raw
        except Exception:
            pass

        # 开始训练
        print("\n🚀 开始训练...", flush=True)
        trainer.train(num_epochs=training_config.num_epochs)

        print("\n✅ 训练完成!", flush=True)

    except Exception as e:
        print("\n❌ 训练过程中发生异常:", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
