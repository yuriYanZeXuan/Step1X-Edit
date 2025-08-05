"""
Ovis模型工具函数
"""

import base64
import os
from io import BytesIO

import torch
from dotenv import load_dotenv
from PIL import Image

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/mnt/data2/huggingface"

load_dotenv()


def encode_image(image: Image.Image) -> str:
    """
    将PIL图像编码为base64字符串

    Args:
        image: PIL图像对象

    Returns:
        base64编码的图像字符串

    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_encoded}"


def load_blank_image(width: int, height: int) -> Image.Image:
    """
    创建空白图像

    Args:
        width: 图像宽度
        height: 图像高度

    Returns:
        空白PIL图像

    """
    pil_image = Image.new("RGB", (width, height), (255, 255, 255)).convert("RGB")
    return pil_image


def build_inputs(
    model,
    text_tokenizer,
    visual_tokenizer,
    prompt: str,
    pil_image: Image.Image,
    target_width: int,
    target_height: int,
):
    """
    构建Ovis模型输入

    Args:
        model: Ovis模型
        text_tokenizer: 文本tokenizer
        visual_tokenizer: 视觉tokenizer
        prompt: 文本提示
        pil_image: 输入图像
        target_width: 目标宽度
        target_height: 目标高度

    Returns:
        模型输入参数

    """
    if pil_image is not None:
        target_size = (int(target_width), int(target_height))
        pil_image, vae_pixel_values, cond_img_ids = model.visual_generator.process_image_aspectratio(
            pil_image, target_size
        )
        cond_img_ids[..., 0] = 1.0
        vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=model.device)
        width = pil_image.width
        height = pil_image.height
        resized_height, resized_width = visual_tokenizer.smart_resize(
            height,
            width,
            max_pixels=visual_tokenizer.image_processor.min_pixels,
        )
        pil_image = pil_image.resize((resized_width, resized_height))
    else:
        vae_pixel_values = None
        cond_img_ids = None

    prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        prompt,
        [pil_image],
        generation_preface=None,
        return_labels=False,
        propagate_exception=False,
        multimodal_type="single_image",
        fix_sample_overall_length_navit=False,
    )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = torch.cat(
            [
                (
                    pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16)
                    if pixel_values is not None
                    else None
                )
            ],
            dim=0,
        )
    if grid_thws is not None:
        grid_thws = torch.cat(
            [(grid_thws.to(device=visual_tokenizer.device) if grid_thws is not None else None)],
            dim=0,
        )
    return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values


def generate_ovis_image(
    model,
    input_img: Image.Image,
    prompt: str,
    steps: int = 50,
    txt_cfg: float = 6.0,
    img_cfg: float = 1.5,
    seed: int = 42,
) -> Image.Image:
    """
    使用Ovis模型生成图像

    Args:
        model: Ovis模型
        input_img: 输入图像
        prompt: 文本提示
        steps: 推理步数
        txt_cfg: 文本引导强度
        img_cfg: 图像引导强度
        seed: 随机种子

    Returns:
        生成的图像

    """
    try:
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        width, height = input_img.size
        height, width = visual_tokenizer.smart_resize(height, width, factor=32)

        gen_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "top_p": None,
            "top_k": None,
            "temperature": None,
            "repetition_penalty": None,
            "eos_token_id": text_tokenizer.eos_token_id,
            "pad_token_id": text_tokenizer.pad_token_id,
            "use_cache": True,
            "height": height,
            "width": width,
            "num_steps": steps,
            "seed": seed,
            "img_cfg": img_cfg,
            "txt_cfg": txt_cfg,
        }

        # 生成无条件图像
        uncond_image = load_blank_image(width, height)
        uncond_prompt = "<image>\nGenerate an image."
        input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs(
            model,
            text_tokenizer,
            visual_tokenizer,
            uncond_prompt,
            uncond_image,
            width,
            height,
        )
        with torch.inference_mode():
            no_both_cond = model.generate_condition(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                grid_thws=grid_thws,
                **gen_kwargs,
            )

        # 生成无文本条件图像
        input_img = input_img.resize((width, height))
        prompt_with_image = "<image>\n" + prompt.strip()
        with torch.inference_mode():
            input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs(
                model,
                text_tokenizer,
                visual_tokenizer,
                uncond_prompt,
                input_img,
                width,
                height,
            )
            no_txt_cond = model.generate_condition(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                grid_thws=grid_thws,
                **gen_kwargs,
            )

        # 生成有条件图像
        input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = build_inputs(
            model,
            text_tokenizer,
            visual_tokenizer,
            prompt_with_image,
            input_img,
            width,
            height,
        )
        with torch.inference_mode():
            cond = model.generate_condition(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                grid_thws=grid_thws,
                **gen_kwargs,
            )
            cond["vae_pixel_values"] = vae_pixel_values
            images = model.generate_img(
                cond=cond,
                no_both_cond=no_both_cond,
                no_txt_cond=no_txt_cond,
                **gen_kwargs,
            )
        return images[0]
    except Exception as e:
        print(f"Ovis生成图像失败: {e}")
        return None
