from typing import List
import base64
from io import BytesIO
from PIL import Image, ImageFile
import requests
import os

# 设置PIL以处理损坏的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_image_to_base64(pil_image, format="PNG"):
    buffered = BytesIO()
    pil_image.save(buffered, format=format)  # Save image to the buffer in the specified format
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')  # Encode the buffer's content to base64
    return img_str

def load_image(image_file):
    """
    加载图像文件，支持多种格式和错误处理
    
    Args:
        image_file: 图像文件路径或URL
        
    Returns:
        PIL.Image: 加载的图像对象
        
    Raises:
        ValueError: 当无法加载图像时
    """
    try:
        if image_file.startswith("http"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # 检查文件是否存在
            if not os.path.exists(image_file):
                raise ValueError(f"Image file does not exist: {image_file}")
            
            # 尝试多种加载方法
            methods = [
                # 方法1: 直接加载
                lambda: Image.open(image_file).convert("RGB"),
                # 方法2: 读取到内存后加载
                lambda: Image.open(BytesIO(open(image_file, 'rb').read())).convert("RGB"),
                # 方法3: 使用PIL的LOAD_TRUNCATED_IMAGES
                lambda: Image.open(image_file).convert("RGB")
            ]
            
            image = None
            last_error = None
            
            for i, method in enumerate(methods):
                try:
                    image = method()
                    break
                except Exception as e:
                    last_error = e
                    print(f"Method {i+1} failed for {image_file}: {e}")
                    continue
            
            if image is None:
                raise ValueError(f"Failed to load image {image_file}: {last_error}")
        
        return image
        
    except Exception as e:
        raise ValueError(f"Error loading image {image_file}: {str(e)}")

def validate_image_file(image_file):
    """
    验证图像文件是否有效
    
    Args:
        image_file: 图像文件路径
        
    Returns:
        bool: 文件是否有效
        str: 错误信息
    """
    try:
        if not os.path.exists(image_file):
            return False, "File does not exist"
        
        # 检查文件大小
        file_size = os.path.getsize(image_file)
        if file_size == 0:
            return False, "File is empty"
        
        # 尝试打开图像
        with open(image_file, 'rb') as f:
            # 读取文件头
            header = f.read(12)
            
            # 检查WebP格式
            if image_file.lower().endswith('.webp'):
                if not header.startswith(b'RIFF'):
                    return False, "Missing RIFF header for WebP file"
            
            # 尝试用PIL打开
            f.seek(0)
            image = Image.open(f)
            image.verify()  # 验证图像完整性
            
            return True, "Valid image file"
            
    except Exception as e:
        return False, f"Error: {str(e)}"


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def merge_images(image_links: List = []):
        """Merge multiple images into one image

        Args:
            image_links (List, optional): List of image links. Defaults to [].

        Returns:
            [type]: [description]
        """
        if len(image_links) == 0:
            return None
        images = load_images(image_links)
        if len(images) == 1:
            return images[0]
        widths, heights = zip(*(i.size for i in images))
        average_height = sum(heights) // len(heights)
        for i, im in enumerate(images):
            # scale in proportion
            images[i] = im.resize((int(im.size[0] * average_height / im.size[1]), average_height))
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new("RGB", (total_width + 10 * (len(images) - 1), max_height))
        x_offset = 0
        for i, im in enumerate(images):
            if i > 0:
                # past a column of 1 pixel starting from x_offset width being black, 8 pixels being white, and 1 pixel being black
                new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
                x_offset += 1
                new_im.paste(Image.new("RGB", (8, max_height), (255, 255, 255)), (x_offset, 0))
                x_offset += 8
                new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
                x_offset += 1
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        return new_im