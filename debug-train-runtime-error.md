# 调试会话：train-runtime-error

- 状态：OPEN
- 症状：运行 `./scripts/envgs/train_render_shitang.sh` 时出现错误。
- 证据来源：`/data/yibo/docs/error.md`

## 待验证假设

1. EnvGS 的 CUDA/OptiX 扩展与当前 RTX 5090、CUDA 或 PyTorch 组合不兼容。
2. 训练脚本传入的配置覆盖项、检查点或输出路径不符合 EasyVolcap 预期。
3. 数据加载、法线文件或图像格式在实际首个训练迭代发生异常。
4. 显存不足或 CUDA 内核执行失败导致训练阶段中断。
5. 环境 Gaussian 的反射路径仅在首个启用反射的迭代暴露配置或实现错误。

## 运行证据

待读取并分析 `error.md`。
