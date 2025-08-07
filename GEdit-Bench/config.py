"""
配置文件，用于管理 GEdit-Bench 生成参数
"""

import os
from typing import Any

# 环境变量配置
ENV_VARS = {
    "GEDIT_BENCH_PATH": "/root/autodl-tmp/hf/hub/datasets--stepfun-ai--GEdit-Bench/snapshots/50766778e2a737474c7e9bdf84cdce82c3ea3f4f",
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HOME": "/root/autodl-tmp/hf",
}

# 设备配置
DEVICE_CONFIG = {
    # "devices": "cuda:0,cuda:1,cuda:2,cuda:3",
    "devices": "cuda:0",
}

# 模型配置
MODEL_CONFIG = {
    # Kontext 模型配置
    "kontext": {
        "flux_model_name": "black-forest-labs/FLUX.1-Kontext-dev",
        "torch_dtype": "bfloat16",
        "model_name": "kontext_dev",
        "weight_path": "/root/autodl-tmp/weights/Kontext",
        "english_only": True,  # Kontext模型默认只处理英文
    },
    # ICEdit 模型配置
    "icedit": {
        "flux_model_name": "black-forest-labs/flux.1-fill-dev",
        "torch_dtype": "bfloat16",
        "model_name": "icedit",
        "english_only": False,  # ICEdit模型处理所有语言
        "lora_weights": "/root/autodl-tmp/weights/icedit",
        'flux_weights': "/root/autodl-tmp/weights/flux"
    },
    # Ovis 模型配置
    "ovis": {
        "model_name": "AIDC-AI/Ovis-U1-3B",
        "weight_path": "/root/autodl-tmp/weights/ovis",
        "torch_dtype": "bfloat16",
        "model_name_output": "ovis",
        "english_only": False,  # Ovis模型处理所有语言
    },
    # stepedit模型配置
    "stepedit": {
        "model_name": "black-forest-labs/flux.1-stepedit-dev",
        "model_path": "/root/autodl-tmp/weights/step1xedit",
        "qwen2vl_model_path": "/root/autodl-tmp/weights/qwen2.5vl",
        "torch_dtype": "bfloat16",
        "model_name_output": "stepedit",
        "version": "v1.0",
        "english_only": False,  # stepedit模型处理所有语言
    },
}

# 生成参数配置
GENERATION_CONFIG = {
    "height": 1024,
    "width": 1024,
    "guidance_scale": 2.5,
    "num_inference_steps": 28,
}

# ICEdit 特定生成参数
ICEDIT_GENERATION_CONFIG = {
    "guidance_scale": 50,  # ICEdit需要更高的guidance_scale
    "num_inference_steps": 28,
    "target_width": 512,  # ICEdit需要512宽度
}

# Ovis 特定生成参数
OVIS_GENERATION_CONFIG = {
    "steps": 50,
    "txt_cfg": 6.0,
    "img_cfg": 1.5,
    "seed": 42,
    "max_new_tokens": 1024,
}

# stepedit 特定生成参数
STEPEDIT_GENERATION_CONFIG = {
    "steps": 28,
    
    "use_teacache": False,
    "teacache_threshold": 0.2,
    "ring_degree": 1,
    "ulysses_degree": 1,
    "cfg_degree": 1,
    
    "use_cache": True,
    "use_taylor_series": True,
    
    "parallel_connector": True,
}

# 结果目录配置
RESULTS_CONFIG = {
    "results_dir": "results",
}

# 任务类型配置（将从数据集中动态读取）
TASK_TYPES = [
    "background_change",
    "color_alter",
    "material_alter",
    "motion_change",
    "ps_human",
    "style_change",
    "subject-add",
    "subject-remove",
    "subject-replace",
    "text_change",
    "tone_transfer",
]

# 语言配置
LANGUAGES = ["cn", "en"]

# 提示词配置（直接使用数据集中的 instruction）
PROMPT_CONFIG = {
    "use_dataset_instruction": True,
    "instruction_field": "instruction",
}

# 负向提示词配置
NEGATIVE_PROMPTS = {
    "base": "low quality, blurry, distorted, deformed, ugly, bad anatomy",
    "detailed": "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, signature, text, logo, oversaturated, overexposed, underexposed",
}

# 默认命令行参数
DEFAULT_ARGS = {
    "model_type": "kontext",
    "model_name": None,  # 将从模型配置中获取
    "devices": None,  # 将从设备配置中获取
    "english_only": None,  # 将从模型配置中获取
    "debug": False,
    "force": False,
    "max_samples": None,
    "max_workers": None,
    "dataset_path": None,
    "results_dir": "results",
}


def get_model_config(model_type: str) -> dict[str, Any]:
    """
    获取指定模型类型的配置

    Args:
        model_type: 模型类型 (kontext/icedit)

    Returns:
        模型配置字典

    """
    if model_type not in MODEL_CONFIG:
        raise ValueError(f"不支持的模型类型: {model_type}")
    return MODEL_CONFIG[model_type]


def get_generation_config(model_type: str) -> dict[str, Any]:
    """
    获取指定模型类型的生成配置

    Args:
        model_type: 模型类型 (kontext/icedit/ovis)

    Returns:
        生成配置字典
    """
    if model_type == "icedit":
        return {**GENERATION_CONFIG, **ICEDIT_GENERATION_CONFIG}
    if model_type == "ovis":
        return {**GENERATION_CONFIG, **OVIS_GENERATION_CONFIG}
    if model_type == "stepedit":
        return {**GENERATION_CONFIG, **STEPEDIT_GENERATION_CONFIG}
    return GENERATION_CONFIG


def get_config(model_type: str = "kontext") -> dict[str, Any]:
    """
    获取完整配置

    Args:
        model_type: 模型类型

    Returns:
        配置字典

    """
    model_config = get_model_config(model_type)
    generation_config = get_generation_config(model_type)

    config = {
        "env_vars": ENV_VARS,
        "device_config": DEVICE_CONFIG,
        "model_config": model_config,
        "generation_config": generation_config,
        "results_config": RESULTS_CONFIG,
        "task_types": TASK_TYPES,
        "languages": LANGUAGES,
        "prompt_config": PROMPT_CONFIG,
        "negative_prompts": NEGATIVE_PROMPTS,
        "model_type": model_type,
    }
    return config


def setup_environment() -> None:
    """设置环境变量"""
    for key, value in ENV_VARS.items():
        if key not in os.environ:
            os.environ[key] = value


def merge_args_with_config(
    args: dict[str, Any], model_type: str
) -> dict[str, Any]:
    """
    合并命令行参数和配置文件

    Args:
        args: 命令行参数字典
        model_type: 模型类型

    Returns:
        合并后的配置字典

    """
    config = get_config(model_type)

    # 合并配置
    merged_config = {
        # 从配置文件加载
        "env_vars": config["env_vars"],
        "model_config": config["model_config"],
        "generation_config": config["generation_config"],
        "results_config": config["results_config"],
        "task_types": config["task_types"],
        "languages": config["languages"],
        "prompt_config": config["prompt_config"],
        "negative_prompts": config["negative_prompts"],
        "model_type": model_type,
        # 从命令行参数加载（优先级更高）
        "model_name": args.get("model_name")
        or config["model_config"]["model_name"],
        "devices": config["device_config"]["devices"].split(","),
        "english_only": args.get("english_only")
        if args.get("english_only") is not None
        else config["model_config"]["english_only"],
        "debug": args.get("debug", False),
        "force": args.get("force", False),
        "max_samples": args.get("max_samples"),
        "max_workers": args.get("max_workers"),
        "dataset_path": args.get("dataset_path"),
        "results_dir": args.get("results_dir", "results"),
    }

    return merged_config
