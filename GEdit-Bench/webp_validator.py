import os
import io
from PIL import Image
import megfile
import argparse

def validate_webp_file(file_path):
    """
    验证WebP文件是否有效
    
    Args:
        file_path: WebP文件路径
        
    Returns:
        bool: 文件是否有效
        str: 错误信息（如果有）
    """
    try:
        # 检查文件是否存在
        if not megfile.exists(file_path):
            return False, "File does not exist"
        
        # 检查文件大小
        file_size = megfile.getsize(file_path)
        if file_size == 0:
            return False, "File is empty"
        
        # 尝试打开图像
        with megfile.smart_open(file_path, 'rb') as f:
            # 读取文件头
            header = f.read(12)
            f.seek(0)
            
            # 检查RIFF头
            if not header.startswith(b'RIFF'):
                return False, "Missing RIFF header"
            
            # 尝试用PIL打开
            image = Image.open(f)
            image.verify()  # 验证图像完整性
            
            return True, "Valid WebP file"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def fix_webp_file(file_path, output_path=None):
    """
    尝试修复损坏的WebP文件
    
    Args:
        file_path: 原始文件路径
        output_path: 输出文件路径（可选）
        
    Returns:
        bool: 是否成功修复
    """
    if output_path is None:
        output_path = file_path
    
    try:
        # 尝试不同的加载方法
        methods = [
            # 方法1: 直接加载
            lambda: Image.open(megfile.smart_open(file_path, 'rb')),
            # 方法2: 读取到内存后加载
            lambda: Image.open(io.BytesIO(megfile.smart_open(file_path, 'rb').read())),
            # 方法3: 转换为JPEG格式
            lambda: Image.open(megfile.smart_open(file_path, 'rb')).convert('RGB')
        ]
        
        for i, method in enumerate(methods):
            try:
                image = method()
                # 保存为PNG格式（更稳定）
                image.save(output_path.replace('.webp', '.png'), 'PNG')
                print(f"Successfully converted {file_path} to PNG format")
                return True
            except Exception as e:
                print(f"Method {i+1} failed: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"Failed to fix file {file_path}: {e}")
        return False

def scan_and_fix_webp_files(directory):
    """
    扫描目录中的所有WebP文件并尝试修复损坏的文件
    
    Args:
        directory: 要扫描的目录
    """
    print(f"Scanning directory: {directory}")
    
    total_files = 0
    valid_files = 0
    fixed_files = 0
    failed_files = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.webp'):
                file_path = os.path.join(root, file)
                total_files += 1
                
                print(f"Checking: {file_path}")
                is_valid, error_msg = validate_webp_file(file_path)
                
                if is_valid:
                    valid_files += 1
                    print(f"  ✓ Valid")
                else:
                    print(f"  ✗ Invalid: {error_msg}")
                    
                    # 尝试修复
                    if fix_webp_file(file_path):
                        fixed_files += 1
                        print(f"  ✓ Fixed")
                    else:
                        failed_files += 1
                        print(f"  ✗ Failed to fix")
    
    print(f"\nSummary:")
    print(f"Total WebP files: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Fixed files: {fixed_files}")
    print(f"Failed files: {failed_files}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebP file validator and fixer")
    parser.add_argument("--path", type=str, required=True, help="File or directory path")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix corrupted files")
    parser.add_argument("--output", type=str, help="Output path for fixed files")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        # 单个文件
        is_valid, error_msg = validate_webp_file(args.path)
        print(f"File: {args.path}")
        print(f"Valid: {is_valid}")
        if not is_valid:
            print(f"Error: {error_msg}")
            if args.fix:
                print("Attempting to fix...")
                fix_webp_file(args.path, args.output)
    else:
        # 目录
        scan_and_fix_webp_files(args.path) 