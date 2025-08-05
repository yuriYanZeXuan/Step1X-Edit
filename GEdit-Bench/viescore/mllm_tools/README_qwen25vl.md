# Qwen2.5-VL API Wrapper

这个模块提供了一个用于调用 Qwen2.5-VL 模型的 API 包装器，仿照 `openai.py` 的接口格式设计。

## 功能特性

- 🖼️ 支持单张和多张图片输入
- 📝 支持文本提示词
- 🔄 支持 API 密钥轮换
- 🛡️ 错误处理和重试机制
- 💬 支持聊天接口
- 🔧 灵活的配置选项

## 安装依赖

```bash
pip install requests pillow
```

## 使用方法

### 1. 基本初始化

```python
from qwen25vl_apieval import Qwen25VL

# 使用单个 API 密钥文件
model = Qwen25VL('path/to/your/api_key.env')

# 使用多个 API 密钥文件进行轮换
model = Qwen25VL(['key1.env', 'key2.env', 'key3.env'])
```

### 2. 单张图片描述

```python
# 准备提示词
prompt = model.prepare_prompt(
    ['https://example.com/image.jpg'], 
    'Describe this image in detail.'
)

# 获取结果
result = model.get_parsed_output(prompt)
print(result)
```

### 3. 多张图片比较

```python
prompt = model.prepare_prompt(
    [
        'https://example.com/image1.jpg',
        'https://example.com/image2.jpg'
    ], 
    'What are the differences between these two images?'
)

result = model.get_parsed_output(prompt)
print(result)
```

### 4. 纯文本对话

```python
prompt = model.prepare_prompt([], 'Hello, how are you?')
result = model.get_parsed_output(prompt)
print(result)
```

### 5. 使用聊天接口

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://example.com/image.jpg"
            },
            {
                "type": "text",
                "text": "What do you see in this image?"
            }
        ]
    }
]

result = model.chat(messages)
print(result)
```

## API 密钥配置

### 1. 创建 API 密钥文件

创建一个文本文件（例如 `secret.env`），将你的 Qwen2.5-VL API 密钥放在第一行：

```
your_api_key_here
```

### 2. 支持的 API 提供商

- **阿里云 DashScope**: 默认配置
- **其他兼容的 API 提供商**: 可以通过修改 `api_base` 参数支持

## 类方法详解

### Qwen25VL 类

#### `__init__(api_key_path, are_images_encoded=False, model_name="qwen2.5-vl-7b-instruct", api_base="https://dashscope.aliyuncs.com/api/v1")`

初始化 Qwen2.5-VL API 包装器。

**参数:**
- `api_key_path` (str or list): API 密钥文件路径或路径列表
- `are_images_encoded` (bool): 图片是否已编码为 base64
- `model_name` (str): 模型名称
- `api_base` (str): API 基础 URL

#### `prepare_prompt(image_links, text_prompt)`

准备发送给 API 的提示词内容。

**参数:**
- `image_links` (List): 图片路径、URL 或 PIL 图像对象列表
- `text_prompt` (str): 文本提示词

**返回:**
- `dict`: 格式化的 API 请求内容

#### `get_parsed_output(prompt)`

发送请求到 Qwen2.5-VL API 并解析响应。

**参数:**
- `prompt` (dict): 准备好的提示词内容

**返回:**
- `str`: 解析后的响应文本

#### `chat(messages, max_tokens=1500, temperature=0.7)`

聊天接口，支持多轮对话。

**参数:**
- `messages` (List[dict]): 消息字典列表
- `max_tokens` (int): 最大生成令牌数
- `temperature` (float): 采样温度

**返回:**
- `str`: 模型响应

#### `update_key(key, load_from_file=True)`

更新 API 密钥。

**参数:**
- `key` (str): 新的 API 密钥或密钥文件路径
- `load_from_file` (bool): 是否从文件加载密钥

## 错误处理

该模块包含完善的错误处理机制：

- **API 密钥轮换**: 当遇到配额限制时自动切换到下一个密钥
- **网络错误**: 处理网络连接问题
- **响应解析**: 处理各种响应格式
- **超时处理**: 设置合理的请求超时时间

## 示例代码

### 完整的图像分析示例

```python
from qwen25vl_apieval import Qwen25VL

# 初始化模型
model = Qwen25VL('secret.env')

# 分析单张图片
def analyze_single_image(image_url, question):
    prompt = model.prepare_prompt([image_url], question)
    result = model.get_parsed_output(prompt)
    return result

# 比较两张图片
def compare_images(image1_url, image2_url):
    prompt = model.prepare_prompt(
        [image1_url, image2_url], 
        'What are the differences between these two images?'
    )
    result = model.get_parsed_output(prompt)
    return result

# 使用示例
image_url = "https://example.com/image.jpg"
description = analyze_single_image(image_url, "Describe this image in detail.")
print(description)
```

### 批量处理示例

```python
import os
from qwen25vl_apieval import Qwen25VL

model = Qwen25VL('secret.env')

def batch_analyze_images(image_folder, question):
    results = {}
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            prompt = model.prepare_prompt([image_path], question)
            result = model.get_parsed_output(prompt)
            results[filename] = result
    
    return results

# 使用示例
results = batch_analyze_images('./images', 'What objects do you see in this image?')
for filename, result in results.items():
    print(f"{filename}: {result}")
```

## 注意事项

1. **API 密钥安全**: 确保 API 密钥文件不被提交到版本控制系统
2. **图片格式**: 支持 JPEG、PNG 等常见格式
3. **网络连接**: 需要稳定的网络连接
4. **配额限制**: 注意 API 调用配额限制
5. **图片大小**: 建议图片大小适中，避免过大的图片影响处理速度

## 故障排除

### 常见问题

1. **API 密钥错误**
   - 检查密钥文件是否存在且格式正确
   - 确认密钥是否有效

2. **网络连接问题**
   - 检查网络连接
   - 确认 API 端点是否可访问

3. **图片加载失败**
   - 检查图片 URL 是否有效
   - 确认图片格式是否支持

4. **响应解析错误**
   - 检查 API 响应格式
   - 确认模型名称是否正确

## 与 openai.py 的兼容性

该模块完全仿照 `openai.py` 的接口设计，主要方法包括：

- `prepare_prompt()`: 准备提示词
- `get_parsed_output()`: 获取解析后的输出
- `update_key()`: 更新 API 密钥

可以直接替换现有的 `openai.py` 调用，无需修改其他代码。 