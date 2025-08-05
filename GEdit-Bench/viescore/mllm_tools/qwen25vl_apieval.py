import base64
import requests
from io import BytesIO
from typing import Union, Optional, Tuple, List
from PIL import Image, ImageOps
import os
import json
import time

def get_api_key(file_path):
    """Read the API key from the first line of the file
    
    Args:
        file_path (str): Path to the API key file
        
    Returns:
        str: API key
    """
    with open(file_path, 'r') as file:
        return file.readline().strip()

def pick_next_item(current_item, item_list):
    """Get the next item in a circular list
    
    Args:
        current_item: Current item in the list
        item_list: List of items
        
    Returns:
        Next item in the list
        
    Raises:
        ValueError: If current item is not in the list
    """
    if current_item not in item_list:
        raise ValueError("Current item is not in the list")
    current_index = item_list.index(current_item)
    next_index = (current_index + 1) % len(item_list)
    return item_list[next_index]

def encode_pil_image(pil_image):
    """Encode a PIL image to base64 string
    
    Args:
        pil_image (PIL.Image): PIL Image object
        
    Returns:
        str: Base64 encoded image string
    """
    image_stream = BytesIO()
    pil_image.save(image_stream, format='JPEG')
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    return base64_image

def load_image(image: Union[str, Image.Image], format: str = "RGB", size: Optional[Tuple] = None) -> Image.Image:
    """
    Load an image from a given path or URL and convert it to a PIL Image.

    Args:
        image (Union[str, Image.Image]): The image path, URL, or a PIL Image object to be loaded.
        format (str, optional): Desired color format of the resulting image. Defaults to "RGB".
        size (Optional[Tuple], optional): Desired size for resizing the image. Defaults to None.

    Returns:
        Image.Image: A PIL Image in the specified format and size.

    Raises:
        ValueError: If the provided image format is not recognized.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = ImageOps.exif_transpose(image)
    image = image.convert(format)
    if size is not None:
        image = image.resize(size, Image.LANCZOS)
    return image

class Qwen25VL():
    """Qwen2.5-VL API wrapper for vision-language tasks
    
    This class provides an interface to interact with Qwen2.5-VL model through API calls.
    It supports both single and multiple image inputs with text prompts.
    """
    
    def __init__(self, api_key_path='secret_t3.env', are_images_encoded=False, 
                 model_name="qwen2.5-vl-72b-instruct", api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"):
        """Initialize Qwen2.5-VL API wrapper
        
        Args:
            api_key_path (str or list): Path to the API key file or list of API key files for rotation
            are_images_encoded (bool): Whether the images are encoded in base64. Defaults to False.
            model_name (str): Model name to use. Defaults to "qwen2.5-vl-7b-instruct".
            api_base (str): API base URL. Defaults to "https://dashscope.aliyuncs.com/api/v1".
        """
        self.multiple_api_keys = False
        self.current_key_file = None
        self.key_lists = None
        
        if isinstance(api_key_path, list):
            self.key_lists = api_key_path
            self.current_key_file = api_key_path[0]
            self.api_key = get_api_key(self.current_key_file)
            self.multiple_api_keys = True
        else:
            self.api_key = get_api_key(api_key_path)
        
        if not self.api_key:
            print("API key not found.")
            exit(1)

        self.api_base = api_base
        self.model_name = model_name
        self.use_encode = are_images_encoded
        self.url = self.api_base

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        """Prepare the prompt content for Qwen2.5-VL API
        
        Args:
            image_links (List): List of image paths, URLs, or PIL images
            text_prompt (str): Text prompt to send with images
            
        Returns:
            dict: Formatted prompt content for API call
        """
        messages = []
        
        # Add text message if provided
        if text_prompt:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            })
        
        # Add images to the message
        if not isinstance(image_links, list):
            image_links = [image_links]
            
        if image_links:
            # If we have images, create a new message or add to existing one
            if not messages:
                messages.append({
                    "role": "user",
                    "content": []
                })
            
            for image_link in image_links:
                image = load_image(image_link)
                if self.use_encode:
                    # Use base64 encoded image
                    image_content = {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{encode_pil_image(image)}"
                    }
                else:
                    # Use image URL or path
                    if isinstance(image_link, str) and (image_link.startswith("http://") or image_link.startswith("https://")):
                        image_content = {
                            "type": "image_url",
                            "image_url": image_link
                        }
                        # print(image_content,"*"*10)
                    else:
                        # For local files, encode them
                        image_content = {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{encode_pil_image(image)}"
                        }
                
                if not messages[-1]["content"]:
                    messages[-1]["content"] = []
                messages[-1]["content"].append(image_content)
            
            # Add text prompt to the same message if we have images
            if text_prompt and messages[-1]["content"]:
                messages[-1]["content"].append({
                    "type": "text",
                    "text": text_prompt
                })
        
        return {
            "messages": messages,
            "model": self.model_name,
            # "input": {
            #     "messages": messages
            # },
            # "parameters": {
            #     "max_tokens": 1500,
            #     "temperature": 0.7,
            #     "top_p": 0.8
            # }
        }

    def get_parsed_output(self, prompt):
        """Send request to Qwen2.5-VL API and parse the response
        
        Args:
            prompt (dict): Prepared prompt content
            
        Returns:
            str: Parsed response text from the model
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(self.url, json=prompt, headers=headers, timeout=60)
            return self.extract_response(response)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return ""

    def extract_response(self, response):
        """Extract and parse the response from Qwen2.5-VL API
        
        Args:
            response: HTTP response object
            
        Returns:
            str: Extracted response text or error message
        """
        try:
            response_data = response.json()
            # print(response_data)
            # exit(0)
            if response.status_code == 200:
                # Success response
                if 'output' in response_data and 'text' in response_data['output']:
                    return response_data['output']['text']
                elif 'choices' in response_data and len(response_data['choices']) > 0:
                    # print(response_data['choices'][0]['message']['content'],"*"*10)
                    # exit(0)
                    return response_data['choices'][0]['message']['content']
                else:
                    print("Unexpected response format")
                    print(response_data)
                    return ""
            else:
                # Error response
                print(response_data)
                exit(0)
                return ""
                
        except json.JSONDecodeError:
            print("Failed to parse JSON response")
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
            return ""
        except Exception as e:
            print(f"Unexpected error: {e}")
            return ""

    def update_key(self, key, load_from_file=True):
        """Update the API key
        
        Args:
            key (str): New API key or path to key file
            load_from_file (bool): Whether to load key from file. Defaults to True.
        """
        if load_from_file:
            self.api_key = get_api_key(key)
        else:
            self.api_key = key

    def chat(self, messages: List[dict], max_tokens: int = 1500, temperature: float = 0.7):
        """Chat interface for Qwen2.5-VL
        
        Args:
            messages (List[dict]): List of message dictionaries with role and content
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Model response
        """
        payload = {
            "model": self.model_name,
            "input": {
                "messages": messages
            },
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.8
            }
        }
        
        return self.get_parsed_output(payload)

if __name__ == "__main__":
    # Example usage
    model = Qwen25VL(
        api_key_path='secret_t3.env', 
        model_name="qwen2.5-vl-72b-instruct",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    
    # Example 1: Single image with text
    prompt = model.prepare_prompt(
        ['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'], 
        'Describe this image in detail.'
    )
    print("Prompt structure:")
    print(json.dumps(prompt, indent=2))
    
    result = model.get_parsed_output(prompt)
    print("Result:")
    print(result)
    
    # Example 2: Multiple images comparison
    prompt2 = model.prepare_prompt(
        [
            'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg',
            'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'
        ], 
        'What are the differences between these two images?'
    )
    
    result2 = model.get_parsed_output(prompt2)
    print("\nComparison result:")
    print(result2) 