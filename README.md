---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: 1ec4f87f54055523d232f718d6fa08f6
    PropagateID: 1ec4f87f54055523d232f718d6fa08f6
    ReservedCode1: 30450220669154b941c442b7b392c3f685067b403c7a1fa1fd0d89c4ad0b7d37d113125b0221009a70bdb6f5f97f5569396dced51311ce41abb5b2045ba3bcdb45dec16bf3072e
    ReservedCode2: 3046022100dac7d9ebb83aae0d7ed24a0f42afd4d5a6adabed0ac6c9873cee9210dc55db2002210097393fe4d24e0f630aa441aa69809c0b80d726b82ecab749726266bdb11b297c
---

# FunASR-GPU-Video-Transcriber

基于 FunASR 的视频转录工具，支持 GPU 加速和 LLM 智能校对。

## 功能特性

- 视频文件自动转录为文字（支持 MP4、MKV、AVI、MOV、FLV、WMV）
- GPU 加速支持（大幅提升转录速度）
- 自动检测 GPU 兼容性，必要时降级到 CPU 模式
- 调用 LM Studio 本地大语言模型进行智能校对
- 多线程并行处理
- 支持批量处理文件夹
- 输出 Markdown 格式转录文档

## 系统要求

### 硬件要求

- **CPU**: 8 核及以上（推荐 16 核）
- **内存**: 16GB 及以上（推荐 64GB）
- **GPU**: NVIDIA 显卡，支持 CUDA（显存 8GB 及以上，推荐 16GB）

### 软件要求

- **Python**: 3.10 - 3.12
- **CUDA**: 12.1 - 12.8（推荐 12.8）
- **PyTorch**: 2.10.0+（推荐，配合 CUDA 12.8）
- **FFmpeg**: 已安装并配置到系统 PATH

### 已测试环境

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12 | 测试通过 |
| PyTorch | 2.10.0+cu128 | RTX 5060 Ti 支持 |
| CUDA | 12.8 | Blackwell 架构支持 |
| GPU | RTX 5060 Ti 16GB | 正常运行 |
| FunASR | 最新版 | SenseVoiceSmall 模型 |

## 快速开始

### 1. 安装依赖

#### 使用 requirements.txt 安装

```bash
pip install -r requirements.txt
```

#### 手动安装

```bash
# 安装 PyTorch（CUDA 12.8 版本）
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# 安装 FunASR
pip install funasr modelscope

# 安装其他依赖
pip install tkinter-table -y
```

### 2. 配置模型路径

编辑 `short-mt-gpu.py`，修改以下配置：

```python
# 模型路径（修改为您的 FunASR 模型路径）
MODEL_DIR = r"I:\AI\APP\FunASR\models\SenseVoiceSmall"

# FFmpeg 路径（修改为您的 FFmpeg 路径）
FFMPEG_PATH = r"I:\AI\APP\FunASR\ffmpeg\bin"
```

### 3. 配置 FFmpeg

确保 FFmpeg 已安装并添加到系统 PATH：

```bash
# 验证 FFmpeg
ffmpeg -version
```

下载 FFmpeg: https://ffmpeg.org/download.html

### 4. 启动 LM Studio（如需校对）

1. 下载并启动 [LM Studio](https://lmstudio.ai/)
2. 加载校对模型（如 Qwen、Llama 等中文模型）
3. 确保 LM Studio 运行在 `http://127.0.0.1:1234`

### 5. 运行程序

```bash
python short-mt-gpu.py
```

## 使用说明

### 主界面

```
┌─────────────────────────────────────────────────┐
│ 视频转录校对工具 - GPU智能加速版                  │
├─────────────────────────────────────────────────┤
│ [✓] 启用GPU加速        GPU设备号: [0]           │
│ 状态: ✓ GPU正常: RTX 5060 Ti                    │
├─────────────────────────────────────────────────┤
│ 输入设置                                          │
│ [选择文件] [选择文件夹]                          │
├─────────────────────────────────────────────────┤
│ LM Studio 设置                                    │
│ IP地址: http://127.0.0.1:1234  [检测连接]       │
│ 状态: 已连接                                     │
└─────────────────────────────────────────────────┘
```

### 操作步骤

1. **配置 GPU**：勾选"启用 GPU 加速"，确认设备号
2. **选择输入**：点击"选择文件"或"选择文件夹"
3. **设置输出**：选择转录结果的保存位置
4. **配置语言**：选择源视频语言（默认自动检测）
5. **启动 LM Studio**（可选）：如需校对，确保 LM Studio 已启动
6. **开始处理**：点击"开始处理"按钮

### GPU/CPU 模式说明

| 模式 | 适用场景 | 处理速度 |
|------|---------|---------|
| GPU 模式 | 有 NVIDIA 显卡 | 快 4-6 倍 |
| CPU 模式 | 无显卡/显卡不支持 | 稳定 |

程序会自动检测 GPU 兼容性，不支持时自动降级到 CPU 模式。

## GPU 兼容性说明

### 支持的 NVIDIA 架构

| 架构 | 系列 | Compute Capability | PyTorch 2.10+ |
|------|------|-------------------|---------------|
| Ampere | RTX 30 系列 | sm_80/sm_86 | ✅ 支持 |
| Ada Lovelace | RTX 40 系列 | sm_90 | ✅ 支持 |
| **Blackwell** | **RTX 50 系列** | **sm_120** | **✅ 支持** |

### RTX 5060 Ti 配置示例

```bash
# 安装 PyTorch 2.10 + CUDA 12.8
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); x=torch.randn(10,10).cuda(); print('GPU OK')"
```

## 项目结构

```
FunASR-GPU-Video-Transcriber/
├── short-mt-gpu.py          # 主程序
├── requirements.txt          # Python 依赖
├── README.md                 # 说明文档
└── docs/
    └── TROUBLESHOOTING.md    # 故障排除指南
```

## 性能对比

| 测试环境 | 视频时长 | GPU 模式 | CPU 模式 |
|---------|---------|---------|---------|
| RTX 5060 Ti 16GB + 16核CPU | 16 分钟 | 1.5-4 分钟 | 8-16 分钟 |
| RTX 3080 10GB + 8核CPU | 16 分钟 | 2-5 分钟 | 10-20 分钟 |

## 常见问题

### Q: 程序提示 "no kernel image is available"

A: PyTorch 版本不支持当前 GPU 架构。请安装支持 Blackwell (sm_120) 的版本：

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

### Q: GPU 模式速度没有明显提升

A: 检查以下事项：
1. GPU 利用率：`nvidia-smi` 查看 GPU 使用情况
2. 批处理大小：在界面上调整"批处理大小(秒)"参数
3. 视频格式：H.264/H.265 编码的视频处理更快

### Q: LM Studio 连接失败

A: 检查以下事项：
1. LM Studio 是否已启动
2. IP 地址是否正确（默认 `http://127.0.0.1:1234`）
3. 防火墙是否阻止了连接

### Q: 转录结果为空

A: 检查以下事项：
1. 视频是否包含音频
2. 音频是否清晰
3. 语言设置是否正确

## 校对提示词

程序使用可自定义的校对提示词，默认为医学文献校对风格。可以在界面上修改提示词以适应不同场景。

提示词模板：
```
你是一位资深的中医文献校对专家。请对以下语音识别文本进行专业校对：
1. **术语修正**：重点修正中药名、穴位名、中医病症名等同音错别字。
2. **标点规范**：调整标点符号，使其符合医学文献规范。
3. **逻辑通顺**：在保持原意的基础上，修正口语化的语法错误。

请直接输出校对后的文本，不要包含任何开场白或解释：

{text}
```

## 许可证

MIT License

## 参考项目

- [FunASR](https://github.com/modelscope/FunASR) - 阿里开源的语音识别工具
- [SenseVoice](https://www.modelscope.cn/models/iic/SenseVoiceSmall) - 阿里云语音识别模型
- [LM Studio](https://lmstudio.ai/) - 本地大语言模型运行工具
