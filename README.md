本仓库用于rm考核

## 项目结构

```text
.
├── src/
│   ├── preview/          # 检测预览、传统视觉、融合、跟踪与预测代码
│   └── assessment/       # 反小陀螺、吊射等考核算法代码
├── models/
│   ├── recommended.onnx  # 运行模型
│   ├── recommended.pt    # 原始权重
│   ├── recommended.json  # 导出参数
│   └── manifest.json     # 模型文件与校验信息
├── run.py                # 预览程序入口
└── requirements.txt      # Python依赖
```
