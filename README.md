# NanoBanana Watermark Remover Portable

一个本地运行的 Nano Banana 图片去水印工具，适合在 Windows 环境下快速处理单张或多张图片。

项目当前提供两种使用方式：

- `WebUI`：打开本地网页界面，批量上传图片并下载处理结果 ZIP
- `CLI`：直接在命令行中处理单张图片

整个处理流程都在本机完成，不依赖云端服务。

## 功能特点

- 支持 `JPG`、`PNG`、`WEBP`
- 支持常规右下角水印清理
- 支持 `1024 x 572` 固定坐标模式
- 支持 `2752 x 1536` 大图模式
- 支持搜索匹配模式
- 支持强力黑边修复
- 支持批量打包下载处理结果

## 项目结构

```text
NanoBanana-Watermark-Remover-Portable/
├─ web_app.py                  # Flask WebUI 后端入口
├─ gemini.py                   # 单图命令行处理脚本
├─ templates/
│  └─ index.html               # WebUI 页面模板
├─ 48.png                      # 小尺寸水印 alpha 参考图
├─ 96.png                      # 大尺寸水印 alpha 参考图
├─ requirements.txt            # 基础依赖
├─ requirements-web.txt        # WebUI 依赖
├─ 一键启动-去水印WebUI.bat     # WebUI 一键启动脚本
└─ 启动-WebUI.bat              # 启动转发脚本
```

## 运行环境

- Windows 10 / 11
- Python 3.10 及以上

## 安装依赖

### 方式一：使用一键启动脚本

直接双击：

- `一键启动-去水印WebUI.bat`

脚本会自动：

1. 检查本地 Python
2. 创建或修复 `.venv`
3. 安装 `requirements-web.txt`
4. 启动本地服务
5. 自动打开浏览器

### 方式二：手动安装

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-web.txt
```

## 启动 WebUI

```powershell
.venv\Scripts\python web_app.py
```

启动后在浏览器打开：

[http://127.0.0.1:5050/](http://127.0.0.1:5050/)

## WebUI 使用方法

1. 打开页面后拖入图片，或点击选择图片
2. 选择处理模式
3. 按需开启“强力黑边清除”
4. 点击开始处理
5. 浏览器自动下载处理后的 ZIP 文件

### 处理模式说明

- `固定坐标`：速度最快，适合常规 Nano Banana 输出
- `搜索匹配`：适合水印位置略有偏移的图片
- `2752 x 1536 大图模式`：适合指定大图尺寸

## 命令行使用方法

处理单张图片：

```powershell
python gemini.py 你的图片路径
```

处理完成后，会在原图旁边生成一个带 `_clean` 后缀的新文件。

例如：

```powershell
python gemini.py demo.png
```

输出文件会是：

```text
demo_clean.png
```

## 依赖列表

当前项目使用的最小依赖如下：

- `flask`
- `numpy`
- `opencv-python`

## 注意事项

- 这是本地工具，图片只会发到本地 Flask 服务，不会上传到远程服务器
- 大批量处理时，ZIP 打包会占用一定内存
- 如果图片尺寸或水印样式与 Nano Banana 常规输出差异较大，建议先拿少量样本测试
- 如果系统里没有 Python，启动脚本会提示先安装 Python 3.10+

## 适合公开发布前再检查的内容

- 是否需要补充项目截图
- 是否保留了需要公开的示例图片
- 是否已经确认 `.gitignore` 不会把本地环境文件一起提交
- 是否已经准备好自己的 GitHub public 仓库地址

## 说明

这个项目当前更偏向实用型桌面辅助工具，重点是本地可用、启动直接、处理路径短。  
如果你准备继续公开维护，后续比较值得补的是：

- 演示截图
- 常见问题说明
- 更多样例对比图
- 简单版本记录
