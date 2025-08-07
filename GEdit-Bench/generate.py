import argparse
import json
import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from config import get_config, merge_args_with_config, setup_environment
from datasets import load_from_disk
from diffusers import DiffusionPipeline
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")
os.environ["HF_HOME"] = os.getenv("HF_HOME")

setup_environment()


class BaseGenerator(ABC):
    """基础图像生成器抽象类"""

    @staticmethod
    def get_available_gpus() -> list[torch.device]:
        """
        获取可用的GPU设备列表

        Returns:
            gpu_devices: 可用的GPU设备列表

        """
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，将使用CPU")
            return [torch.device("cpu")]

        gpu_count = torch.cuda.device_count()
        gpu_devices = [torch.device(f"cuda:{i}") for i in range(gpu_count)]
        logger.info(f"检测到 {gpu_count} 个GPU设备: {gpu_devices}")
        return gpu_devices

    def __init__(
        self,
        device: torch.device,
        results_dir: str = "results",
        model_name: str = "base",
        config: dict[str, Any] | None = None,
    ):
        """
        初始化基础生成器

        Args:
            device: 模型使用的设备
            results_dir: 结果保存目录
            model_name: 模型名称
            config: 配置字典，如果为None则使用默认配置

        """
        self.device = device
        self.results_dir = Path(results_dir)
        self.model_name = model_name

        # 加载配置
        if config is not None:
            self.config = config
        else:
            self.config = get_config()

        # 初始化模型（由子类实现）
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self) -> None:
        """初始化模型，由子类实现"""

    @abstractmethod
    def _generate_image(
        self,
        original_image: Image.Image,
        instruction: str,
        task_type: str,
        instruction_language: str,
        **kwargs,
    ) -> Image.Image:
        """
        生成编辑后的图像，由子类实现

        Args:
            original_image: 原始图像
            instruction: 编辑指令
            task_type: 任务类型
            instruction_language: 指令语言
            **kwargs: 其他参数

        Returns:
            edited_image: 编辑后的图像

        """

    def _create_results_structure(self, task_types: list[str]) -> None:
        """创建结果目录结构"""
        # 按照 EVAL.md 中的要求创建目录结构
        # results/{method_name}/fullset/{edit_task}/{cn|en}/
        base_dir = self.results_dir / self.model_name / "fullset"

        # 使用配置文件中的语言
        languages = self.config["languages"]

        # 为每个任务类型创建目录
        for task_type in task_types:
            for lang in languages:
                task_dir = base_dir / task_type / lang
                task_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"创建目录: {task_dir}")

    def _save_image(
        self,
        image: Image.Image,
        task_type: str,
        instruction_language: str,
        key: str,
        name: str = "",
    ) -> str:
        """
        保存生成的图像

        Args:
            image: 生成的图像
            task_type: 任务类型
            instruction_language: 指令语言
            key: 图像键值
            name: 自定义文件名

        Returns:
            saved_path: 保存路径

        """
        # 构建保存路径
        save_dir = (
            self.results_dir
            / self.model_name
            / "fullset"
            / task_type
            / instruction_language
        )

        # 确保目录存在
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存图像
        if name:
            save_path = save_dir / f"{name}.png"
        else:
            save_path = save_dir / f"{key}.png"
        image.save(save_path)

        return str(save_path)

    def _check_image_exists(
        self, task_type: str, instruction_language: str, key: str
    ) -> bool:
        """
        检查图像是否已经生成

        Args:
            task_type: 任务类型
            instruction_language: 指令语言
            key: 图像键值

        Returns:
            exists: 图像是否已存在

        """
        image_path = (
            self.results_dir
            / self.model_name
            / "fullset"
            / task_type
            / instruction_language
            / f"{key}.png"
        )
        return image_path.exists()

    def generate_single_image(
        self,
        original_image: Image.Image,
        instruction: str,
        task_type: str,
        instruction_language: str,
        key: str,
        force: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> str | None:
        """
        生成单张编辑后的图像

        Args:
            original_image: 原始图像
            instruction: 编辑指令
            task_type: 任务类型
            instruction_language: 指令语言
            key: 图像键值
            force: 是否强制重新生成
            debug: 是否启用调试模式
            **kwargs: 其他参数

        Returns:
            saved_path: 保存的图像路径，失败时返回 None

        """
        try:
            # 检查图像是否已存在
            if not force and self._check_image_exists(
                task_type, instruction_language, key
            ):
                logger.info(f"图像 {key} 已存在，跳过生成")
                return str(
                    self.results_dir
                    / self.model_name
                    / "fullset"
                    / task_type
                    / instruction_language
                    / f"{key}.png"
                )
            # if debug:
            #     instruction = "add a cat to the image"
            logger.info(f"处理图像 {key}: {instruction}")

            # 生成图像
            output_image = self._generate_image(
                original_image=original_image,
                instruction=instruction,
                task_type=task_type,
                instruction_language=instruction_language,
                **kwargs,
            )

            if output_image is None:
                logger.error(f"生成图像 {key} 失败")
                return None

            # 保存图像
            logger.info(f"保存模型生成的输出图像，尺寸: {output_image.size}")
            saved_path = self._save_image(
                output_image,
                task_type,
                instruction_language,
                key,
                name=key,
            )

            # Debug模式：输出详细信息
            if debug:
                # save the original image
                original_image_path = self._save_image(
                    original_image,
                    task_type,
                    instruction_language,
                    key,
                    name=f"{key}_original",
                )
                logger.info("=" * 50)
                logger.info("DEBUG 模式 - 图像生成详情:")
                logger.info(f"Key: {key}")
                logger.info(f"Task Type: {task_type}")
                logger.info(f"Language: {instruction_language}")
                logger.info(f"Instruction: {instruction}")
                logger.info(f"Input Image Size: {original_image.size}")
                logger.info(f"Output Image Size: {output_image.size}")
                logger.info(f"Saved Original Image: {original_image_path}")
                logger.info(f"Saved Path: {saved_path}")
                logger.info("=" * 50)
                # Debug模式下只处理一个样本就停止
                raise SystemExit("Debug模式：已处理一个样本，程序停止")

            logger.info(f"图像 {key} 生成完成: {saved_path}")
            return saved_path

        except SystemExit:
            # 重新抛出SystemExit，让程序停止
            raise
        except Exception as e:
            logger.error(f"生成图像 {key} 时出错: {e!s}")
            raise
            # return None

    def generate_all_images(
        self,
        dataset_path: str | None = None,
        max_samples: int | None = None,
        force: bool = False,
        debug: bool = False,
        english_only: bool = False,
        **kwargs,
    ) -> dict[str, list[str]]:
        """
        为整个数据集生成编辑后的图像

        Args:
            dataset_path: 数据集路径，如果为 None 则使用环境变量
            max_samples: 最大处理样本数，用于测试
            force: 是否强制重新生成已存在的图像
            debug: 是否启用调试模式
            english_only: 是否只处理英文任务
            **kwargs: 其他参数

        Returns:
            results: 按任务类型分组的生成结果

        """
        # 加载数据集
        if dataset_path is None:
            dataset_path = os.getenv("GEDIT_BENCH_PATH")

        if not dataset_path:
            raise ValueError(
                "数据集路径未设置，请设置 GEDIT_BENCH_PATH 环境变量"
            )

        logger.info(f"正在加载数据集: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        logger.info(f"数据集加载完成，共 {len(dataset)} 个样本")

        # 从数据集中获取所有任务类型
        task_types = self.config["task_types"]
        logger.info(f"发现任务类型: {task_types}")

        # 显示english_only模式状态
        if english_only:
            logger.info("启用english_only模式：将跳过中文任务，只处理英文任务")

        # 创建结果目录结构
        self._create_results_structure(task_types)

        # 按任务类型分组
        results: dict[str, list[str]] = {}

        # 处理每个样本
        for i, item in enumerate(tqdm(dataset, desc="生成编辑图像")):
            if max_samples and i >= max_samples:
                break

            task_type = item["task_type"]
            instruction = item["instruction"]
            instruction_language = item["instruction_language"]
            key = item["key"]
            original_image = item["input_image"]

            # 对于kontext模型，如果启用english_only，则跳过中文任务
            if english_only and instruction_language == "cn":
                logger.debug(f"跳过中文任务: {key} ({instruction_language})")
                continue

            # 生成图像
            saved_path = self.generate_single_image(
                original_image=original_image,
                instruction=instruction,
                task_type=task_type,
                instruction_language=instruction_language,
                key=key,
                force=force,
                debug=debug,
                **kwargs,
            )

            if saved_path:
                if task_type not in results:
                    results[task_type] = []
                results[task_type].append(saved_path)

        # 输出统计信息
        logger.info("生成完成！统计信息:")
        for task_type, paths in results.items():
            logger.info(f"  {task_type}: {len(paths)} 张图像")

        return results


class KontextEvalGenerator(BaseGenerator):
    """使用 FLUX Kontext-dev 模型进行图像编辑生成的评估器"""

    def _initialize_model(self) -> None:
        from pipeline_flux_kontext import FluxKontextPipeline
        """初始化 FLUX Kontext-dev 模型"""
        logger.info("正在初始化 FLUX Kontext-dev 模型...")
        self.flux = FluxKontextPipeline.from_pretrained(
            self.config["model_config"]["weight_path"],
        )
        self.flux.to_empty(self.device)
        logger.info("FLUX 模型初始化完成")

    def _get_prompt_for_task(
        self, instruction: str, task_type: str, instruction_language: str
    ) -> tuple[str, str]:
        """
        获取提示词（直接使用数据集中的 instruction）

        Args:
            instruction: 原始指令（直接来自数据集）
            task_type: 任务类型
            instruction_language: 指令语言 (cn/en)

        Returns:
            positive_prompt: 正向提示词
            negative_prompt: 负向提示词

        """
        # 直接使用数据集中的 instruction 作为正向提示词
        positive_prompt = instruction

        # 获取负向提示词
        negative_prompts = self.config["negative_prompts"]
        negative_prompt = negative_prompts["base"]

        return positive_prompt, negative_prompt

    def _generate_image(
        self,
        original_image: Image.Image,
        instruction: str,
        task_type: str,
        instruction_language: str,
        **kwargs,
    ) -> Image.Image:
        """
        使用 FLUX Kontext-dev 生成编辑后的图像

        Args:
            original_image: 原始图像
            instruction: 编辑指令
            task_type: 任务类型
            instruction_language: 指令语言
            **kwargs: 其他参数

        Returns:
            edited_image: 编辑后的图像

        """
        # 生成提示词
        positive_prompt, negative_prompt = self._get_prompt_for_task(
            instruction, task_type, instruction_language
        )

        # 使用配置文件中的生成参数
        gen_config = self.config["generation_config"]

        # 使用 FLUX 模型生成图像
        with torch.no_grad():
            output_image = self.flux(
                image=original_image,
                prompt=positive_prompt,
                height=gen_config["height"],
                width=gen_config["width"],
                # guidance_scale=gen_config["guidance_scale"],
                # num_inference_steps=gen_config["num_inference_steps"],
            ).images[0]

        return output_image


class ICEditGenerator(BaseGenerator):
    """使用 FLUX ICEdit 模型进行图像编辑生成的评估器"""

    def _initialize_model(self) -> None:
        """初始化 FLUX ICEdit 模型"""
        logger.info("正在初始化 FLUX ICEdit 模型...")
        from diffusers import FluxFillPipeline

        self.flux = FluxFillPipeline.from_pretrained(
            self.config["model_config"]["flux_weights"],
            torch_dtype=getattr(
                torch, self.config["model_config"]["torch_dtype"]
            ),
        )
        self.flux.load_lora_weights(self.config["model_config"]["lora_weights"])
        self.flux.to_empty(self.device)
        logger.info("FLUX ICEdit 模型初始化完成")

    def _generate_image(
        self,
        original_image: Image.Image,
        instruction: str,
        task_type: str,
        instruction_language: str,
        **kwargs,
    ) -> Image.Image:
        """
        使用ICEdit生成图像

        Args:
            original_image: 原始图像
            instruction: 编辑指令
            task_type: 任务类型
            instruction_language: 指令语言
            **kwargs: 其他参数

        Returns:
            edited_image: 编辑后的图像

        """
        try:
            # 准备ICEdit输入
            # 调整图像尺寸为目标宽度
            target_width = self.config["generation_config"]["target_width"]
            if original_image.size[0] != target_width:
                new_width = target_width
                scale = new_width / original_image.size[0]
                new_height = int(original_image.size[1] * scale)
                new_height = (new_height // 8) * 8
                resized_image = original_image.resize((new_width, new_height))
            else:
                resized_image = original_image

            # 构建ICEdit指令
            instruction_text = f"A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}"

            # 创建组合图像和mask
            width, height = resized_image.size
            combined_image = Image.new("RGB", (width * 2, height))
            combined_image.paste(resized_image, (0, 0))
            combined_image.paste(resized_image, (width, 0))

            import numpy as np

            # 创建mask - 只mask右侧部分（需要编辑的区域）
            mask_array = np.zeros((height, width * 2), dtype=np.uint8)
            mask_array[:, width:] = 255
            mask = Image.fromarray(mask_array)

            # 使用ICEdit生成图像
            gen_config = self.config["generation_config"]

            # 添加调试信息
            logger.debug(
                f"ICEdit参数: guidance_scale={gen_config['guidance_scale']}, steps={gen_config['num_inference_steps']}"
            )
            logger.debug(f"指令: {instruction_text}")
            logger.debug(
                f"图像尺寸: {width}x{height}, 组合尺寸: {width * 2}x{height}"
            )

            output_image = self.flux(
                prompt=instruction_text,
                image=combined_image,
                mask_image=mask,
                height=height,
                width=width * 2,
                guidance_scale=gen_config["guidance_scale"],
                num_inference_steps=gen_config["num_inference_steps"],
                generator=torch.Generator("cpu").manual_seed(42),
            ).images[0]

            # 裁剪出右侧部分（修改后的图像）
            output_image = output_image.crop((width, 0, width * 2, height))

            # 检查输出图像是否与输入相同
            if np.array_equal(np.array(output_image), np.array(resized_image)):
                logger.warning(
                    "ICEdit生成的图像与原始图像相同，可能需要调整参数"
                )

            return output_image
        except Exception as e:
            logger.error(f"ICEdit生成图像失败: {e}")
            return None


class OvisGenerator(BaseGenerator):
    """使用 Ovis 模型进行图像编辑生成的评估器"""

    def _initialize_model(self) -> None:
        """初始化 Ovis 模型"""
        logger.info("正在初始化 Ovis 模型...")
        from transformers import AutoModelForCausalLM

        self.ovis = AutoModelForCausalLM.from_pretrained(
            self.config["model_config"]["model_name"] \
                if 'weight_path' not in self.config["model_config"] \
                else self.config["model_config"]["weight_path"],
            torch_dtype=getattr(
                torch, self.config["model_config"]["torch_dtype"]
            ),
            trust_remote_code=True,
        )
        self.ovis = self.ovis.eval().to_empty(device=self.device)
        self.ovis = self.ovis.to(
            getattr(torch, self.config["model_config"]["torch_dtype"])
        )
        logger.info("Ovis 模型初始化完成")

    def _generate_image(
        self,
        original_image: Image.Image,
        instruction: str,
        task_type: str,
        instruction_language: str,
        **kwargs,
    ) -> Image.Image:
        """
        使用Ovis生成图像

        Args:
            original_image: 原始图像
            instruction: 编辑指令
            task_type: 任务类型
            instruction_language: 指令语言
            **kwargs: 其他参数

        Returns:
            edited_image: 编辑后的图像

        """
        try:
            # 导入Ovis相关函数
            from ovis_utils import generate_ovis_image

            # 构建Ovis指令
            prompt = f"Edit the image to {instruction}"

            # 获取生成参数
            gen_config = self.config["generation_config"]

            # 添加调试信息
            logger.debug(
                f"Ovis参数: steps={gen_config['steps']}, txt_cfg={gen_config['txt_cfg']}, img_cfg={gen_config['img_cfg']}"
            )
            logger.debug(f"指令: {prompt}")
            logger.debug(f"图像尺寸: {original_image.size}")

            # 使用Ovis生成图像
            output_image = generate_ovis_image(
                model=self.ovis,
                input_img=original_image,
                prompt=prompt,
                steps=gen_config["steps"],
                txt_cfg=gen_config["txt_cfg"],
                img_cfg=gen_config["img_cfg"],
                seed=gen_config["seed"],
            )

            if output_image is None:
                logger.error("Ovis生成图像失败，返回None")
                return None

            # 检查输出图像是否与输入相同
            import numpy as np

            if np.array_equal(np.array(output_image), np.array(original_image)):
                logger.warning("Ovis生成的图像与原始图像相同，可能需要调整参数")

            return output_image
        except Exception as e:
            logger.error(f"Ovis生成图像失败: {e}")
            return None


class StepEditGenerator(BaseGenerator):
    """使用 Step1X-Edit/inference.py 的 ImageGenerator 进行图像编辑生成"""

    def _initialize_model(
        self,
        device: torch.device = torch.device("cuda:0"),
        results_dir: str = "results",
        model_name: str = "stepedit",
        config: dict[str, Any] = None,
    ):
        """初始化 StepEdit 模型"""
        logger.info("正在初始化 StepEdit 模型...")
        import sys
        import os
        # 添加父目录到Python路径以支持导入
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from inference import ImageGenerator

        self.device = device
        self.results_dir = Path(results_dir)
        self.config = config or get_config('stepedit')

        # 读取模型路径和参数
        model_config = self.config.get("model_config", {})
        model_path = model_config.get("model_path", "/root/autodl-tmp/weights/step1xedit")
        qwen2vl_model_path = model_config.get("qwen2vl_model_path", "/root/autodl-tmp/weights/qwen2.5vl")
        version = model_config.get("version", "v1.0")
        logger.info(f"model_config:{model_config}")
        
        # gen_config用于设置cache相关参数
        gen_config = self.config.get("generation_config", {})
        logger.info(f"gen_config:{gen_config}")
        
        size_level = self.config.get("size_level", 512)
        quantized = self.config.get("quantized", False)
        offload = self.config.get("offload", False)
        lora = self.config.get("lora", None)
        max_length = self.config.get("max_length", 640)

        if version == 'v1.0':
            ckpt_name = 'step1x-edit-i1258.safetensors'
        elif version == 'v1.1':
            ckpt_name = 'step1x-edit-v1p1-official.safetensors'
        else:
            raise ValueError(f"未知的StepEdit版本: {version}")

        # 初始化ImageGenerator
        self.pipe = ImageGenerator(
            ae_path=str(Path(model_path) / 'vae.safetensors'),
            dit_path=str(Path(model_path) / ckpt_name),
            qwen2vl_model_path=qwen2vl_model_path,
            max_length=max_length,
            quantized=quantized,
            offload=offload,
            mode="flash",
            version=version,
            **gen_config
        )
        if gen_config.get("use_teacache", False): 
            self.pipe.dit.__class__.enable_teacache = True
            self.pipe.dit.__class__.cnt = 0
            self.pipe.dit.__class__.num_steps = gen_config.get("steps", 28)
            self.pipe.dit.__class__.rel_l1_thresh = gen_config.get("teacache_threshold", 0.2)
            self.pipe.dit.__class__.accumulated_rel_l1_distance = 0
            self.pipe.dit.__class__.previous_modulated_input = None
            self.pipe.dit.__class__.previous_residual = None
            from modules.multigpu import teacache_transformer
            teacache_transformer(self.pipe)
            logger.info("teacache_transformer完成")

    def _generate_image(
        self,
        original_image: Image.Image,
        instruction: str,
        task_type: str,
        instruction_language: str,
        **kwargs,
    ) -> Image.Image | None:
        """
        使用StepEdit模型进行图像编辑

        Args:
            original_image: 输入的原始图像（PIL.Image）
            instruction: 编辑指令（字符串）
            seed: 随机种子
            negative_prompt: 负面提示词
            num_steps: 采样步数
            cfg_guidance: CFG引导强度
            size_level: 输入分辨率
            height: 输出高度
            width: 输出宽度

        Returns:
            编辑后的图像（PIL.Image），失败时返回None
        """
        try:
            prompt = instruction
            gen_config = self.config.get("generation_config", {})
            num_steps = gen_config.get("steps", 28)
            cfg_guidance = gen_config.get("cfg_guidance", 6.0)
            size_level = gen_config.get("size_level", 512)
            height = gen_config.get("height", 1024)
            width = gen_config.get("width", 1024)
            
            # 从kwargs中获取参数，如果没有则使用默认值
            negative_prompt = kwargs.get("negative_prompt", "")
            seed = kwargs.get("seed", 42)

            # StepEdit 只支持单张图片
            ref_images = original_image

            # 调用ImageGenerator的generate_image方法
            images = self.pipe.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ref_images=ref_images,
                num_samples=1,
                num_steps=num_steps,
                cfg_guidance=cfg_guidance,
                seed=seed,
                show_progress=False,
                size_level=size_level,
                height=height,
                width=width,
            )
            if not images or images[0] is None:
                logger.error("StepEdit生成图像失败，返回None")
                return None

            # 检查输出图像是否与输入相同
            if np.array_equal(np.array(images[0]), np.array(original_image)):
                logger.warning("StepEdit生成的图像与原始图像相同，可能需要调整参数")

            return images[0]
        except Exception as e:
            logger.error(f"StepEdit生成图像失败: {e}")
            raise
            # return None


class ParallelGenerator:
    """多GPU并行图像生成器"""

    def __init__(
        self,
        generator_class: type[BaseGenerator],
        results_dir: str = "results",
        model_name: str = "base",
        max_workers: int | None = None,
        devices: list[torch.device] | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        初始化并行生成器

        Args:
            generator_class: 生成器类（继承自BaseGenerator）
            results_dir: 结果保存目录
            model_name: 模型名称
            max_workers: 最大工作进程数，如果为None则使用GPU数量
            devices: 用户指定的设备列表，如果为None则自动检测
            config: 配置字典

        """
        self.generator_class = generator_class
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        self.config = config or get_config()

        # 获取可用的GPU设备
        if devices is None:
            self.gpu_devices = BaseGenerator.get_available_gpus()
        else:
            self.gpu_devices = devices
            logger.info(f"使用用户指定的设备: {devices}")

        self.max_workers = max_workers or len(self.gpu_devices)

        logger.info(f"初始化并行生成器，使用 {self.max_workers} 个工作进程")

        # 初始化每个GPU上的模型
        self.generators = []
        for i, device in enumerate(self.gpu_devices[: self.max_workers]):
            generator = generator_class(
                device=device,
                results_dir=str(self.results_dir),
                model_name=self.model_name,
                config=self.config,
            )
            self.generators.append(generator)
            logger.info(
                f"在设备 {device} 上初始化模型 {i + 1}/{self.max_workers}"
            )

    def _create_results_structure(self, task_types: list[str]) -> None:
        """创建结果目录结构"""
        # 使用第一个生成器创建目录结构
        if self.generators:
            self.generators[0]._create_results_structure(task_types)

    def _check_image_exists(
        self, task_type: str, instruction_language: str, key: str
    ) -> bool:
        """检查图像是否已经生成"""
        image_path = (
            self.results_dir
            / self.model_name
            / "fullset"
            / task_type
            / instruction_language
            / f"{key}.png"
        )
        return image_path.exists()

    def _process_single_item(
        self,
        item: dict,
        generator_id: int,
        force: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> tuple[str, str] | None:
        """
        处理单个数据项

        Args:
            item: 数据项
            generator_id: 生成器ID
            force: 是否强制重新生成
            debug: 是否启用调试模式
            **kwargs: 其他参数

        Returns:
            (key, saved_path) 或 None

        """
        try:
            generator = self.generators[generator_id]

            task_type = item["task_type"]
            instruction = item["instruction"]
            instruction_language = item["instruction_language"]
            key = item["key"]
            original_image = item["input_image"]

            # 生成图像
            saved_path = generator.generate_single_image(
                original_image=original_image,
                instruction=instruction,
                task_type=task_type,
                instruction_language=instruction_language,
                key=key,
                force=force,
                debug=debug,
                **kwargs,
            )

            if saved_path:
                return key, saved_path
            return None

        except Exception as e:
            logger.error(f"处理图像 {item.get('key', 'unknown')} 时出错: {e}")
            return None

    def generate_all_images(
        self,
        dataset_path: str | None = None,
        max_samples: int | None = None,
        force: bool = False,
        debug: bool = False,
        english_only: bool = False,
        **kwargs,
    ) -> dict[str, list[str]]:
        """
        并行生成所有图像

        Args:
            dataset_path: 数据集路径
            max_samples: 最大处理样本数
            force: 是否强制重新生成
            debug: 是否启用调试模式
            english_only: 是否只处理英文任务
            **kwargs: 其他参数

        Returns:
            results: 按任务类型分组的生成结果

        """
        # 加载数据集
        if dataset_path is None:
            dataset_path = os.getenv("GEDIT_BENCH_PATH")

        if not dataset_path:
            raise ValueError(
                "数据集路径未设置，请设置 GEDIT_BENCH_PATH 环境变量"
            )

        logger.info(f"正在加载数据集: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        logger.info(f"数据集加载完成，共 {len(dataset)} 个样本")

        # 从数据集中获取所有任务类型
        task_types = self.config["task_types"]
        logger.info(f"发现任务类型: {task_types}")

        # 显示english_only模式状态
        if english_only:
            logger.info("启用english_only模式：将跳过中文任务，只处理英文任务")

        # 创建结果目录结构
        self._create_results_structure(task_types)

        # 按任务类型分组
        results: dict[str, list[str]] = {}

        # 准备数据项列表
        items_to_process = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            # 对于kontext模型，如果启用english_only，则跳过中文任务
            if english_only and item["instruction_language"] == "cn":
                logger.debug(
                    f"跳过中文任务: {item['key']} ({item['instruction_language']})"
                )
                continue

            items_to_process.append(item)

        logger.info(
            f"准备处理 {len(items_to_process)} 个样本（已过滤中文任务）"
        )

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {}
            for i, item in enumerate(items_to_process):
                generator_id = i % len(self.generators)
                future = executor.submit(
                    self._process_single_item,
                    item,
                    generator_id,
                    force,
                    debug,
                    **kwargs,
                )
                future_to_item[future] = item

            # 处理完成的任务
            for future in tqdm(
                as_completed(future_to_item),
                total=len(future_to_item),
                desc="并行生成图像",
            ):
                result = future.result()
                if result:
                    key, saved_path = result
                    task_type = future_to_item[future]["task_type"]
                    if task_type not in results:
                        results[task_type] = []
                    results[task_type].append(saved_path)

        # 输出统计信息
        logger.info("并行生成完成！统计信息:")
        for task_type, paths in results.items():
            logger.info(f"  {task_type}: {len(paths)} 张图像")

        return results


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="使用 FLUX 模型生成 GEdit-Bench 图像"
    )

    # 核心参数
    parser.add_argument(
        "--model_type",
        type=str,
        default="kontext",
        choices=["stepedit","kontext", "icedit", "ovis"],
        help="模型类型:stepedit、kontext、icedit 或 ovis",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="模型名称（可选，默认从配置文件获取）",
    )
    parser.add_argument(
        "--english_only",
        action="store_true",
        default=None,
        help="只处理英文任务（可选，默认从配置文件获取）",
    )

    # 生成控制参数
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式，只处理一个样本并输出详细信息",
    )
    parser.add_argument(
        "--force", action="store_true", help="强制重新生成已存在的图像"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="最大处理样本数，用于测试"
    )

    # 并行处理参数
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="最大工作进程数（并行模式）",
    )

    # 路径参数
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="数据集路径，如果未指定则使用环境变量",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results", help="结果保存目录"
    )

    args = parser.parse_args()

    # 设置环境
    setup_environment()

    # 将args转换为字典
    args_dict = vars(args)

    # 合并配置
    config = merge_args_with_config(args_dict, args.model_type)

    # 根据模型类型选择生成器类
    if args.model_type == "kontext":
        generator_class = KontextEvalGenerator
    elif args.model_type == "icedit":
        generator_class = ICEditGenerator
    elif args.model_type == "ovis":
        generator_class = OvisGenerator
    elif args.model_type == "stepedit":
        generator_class = StepEditGenerator
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

    # 处理设备配置
    if config["devices"] and len(config["devices"]) > 1:
        # 用户指定了多个设备，使用并行模式
        user_devices = [torch.device(device) for device in config["devices"]]
        logger.info(f"用户指定的设备列表: {user_devices}")

        # 检查所有设备是否可用
        for device in user_devices:
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise ValueError("CUDA不可用，但用户指定了CUDA设备")
                device_id = device.index
                if device_id >= torch.cuda.device_count():
                    raise ValueError(
                        f"指定的CUDA设备 {device_id} 不存在，可用设备数: {torch.cuda.device_count()}"
                    )

        # 使用并行生成器
        logger.info("使用用户指定的设备进行并行生成")
        generator = ParallelGenerator(
            generator_class=generator_class,
            results_dir=config["results_dir"],
            model_name=config["model_name"],
            max_workers=config["max_workers"],
            devices=user_devices,
            config=config,
        )
    else:
        # 使用单个设备
        print(config["devices"])
        user_device = torch.device(config["devices"][0])
        logger.info(f"使用设备: {user_device}")

        # 检查设备是否可用
        if user_device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA不可用，但用户指定了CUDA设备")
            device_id = user_device.index
            if device_id >= torch.cuda.device_count():
                raise ValueError(
                    f"指定的CUDA设备 {device_id} 不存在，可用设备数: {torch.cuda.device_count()}"
                )

        logger.info("使用单GPU生成模式")
        generator = generator_class(
            device=user_device,
            results_dir=config["results_dir"],
            model_name=config["model_name"],
            config=config,
        )

    if config["debug"]:
        logger.setLevel(logging.DEBUG)

    # 生成所有图像
    results = generator.generate_all_images(
        dataset_path=config["dataset_path"],
        max_samples=config["max_samples"],
        force=config["force"],
        debug=config["debug"],
        english_only=config["english_only"],
    )

    logger.info("所有图像生成完成！")

    # 保存配置
    config_save_path = (
        Path(config["results_dir"]) / config["model_type"] / "config.json"
    )
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"配置已保存到: {config_save_path}")


if __name__ == "__main__":
    main()
