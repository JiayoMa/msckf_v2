import os
import sys

def find_all_py_files(root_dir):
    """递归查找所有Python文件"""
    py_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 跳过虚拟环境目录（可选）
        if 'venv' in dirpath or '.env' in dirpath or '__pycache__' in dirpath:
            continue
            
        for filename in filenames:
            if filename.endswith('.py'):
                full_path = os.path.join(dirpath, filename)
                py_files.append(full_path)
    
    return py_files

def read_file_content(filepath):
    """读取文件内容，并添加文件信息"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建文件头信息
        file_info = f"\n{'='*80}\n"
        file_info += f"文件路径: {filepath}\n"
        file_info += f"文件大小: {os.path.getsize(filepath)} 字节\n"
        file_info += f"{'='*80}\n\n"
        
        return file_info + content
    except Exception as e:
        return f"\n{'='*80}\n错误: 无法读取文件 {filepath}\n错误信息: {str(e)}\n{'='*80}\n\n"

def combine_py_files(output_file='combined_code.txt'):
    """将所有的Python文件内容合并到一个文件中"""
    # 获取当前目录
    current_dir = os.getcwd()
    
    print(f"正在扫描目录: {current_dir}")
    
    # 查找所有Python文件
    py_files = find_all_py_files(current_dir)
    
    if not py_files:
        print("没有找到任何Python文件 (.py)")
        return
    
    print(f"找到 {len(py_files)} 个Python文件")
    
    # 创建输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 写入汇总信息
        outfile.write(f"Python代码合并文件\n")
        outfile.write(f"生成时间: {os.path.getctime(output_file) if os.path.exists(output_file) else 'N/A'}\n")
        outfile.write(f"扫描目录: {current_dir}\n")
        outfile.write(f"文件总数: {len(py_files)}\n")
        outfile.write(f"{'='*80}\n\n")
        
        # 写入每个文件的内容
        for i, filepath in enumerate(py_files, 1):
            print(f"正在处理 ({i}/{len(py_files)}): {os.path.relpath(filepath, current_dir)}")
            
            content = read_file_content(filepath)
            outfile.write(content)
            outfile.write("\n\n")  # 文件之间添加空行
    
    print(f"\n完成! 所有代码已保存到: {output_file}")
    print(f"总文件数: {len(py_files)}")
    
    # 显示一些统计信息
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            total_size = len(f.read())
        print(f"输出文件大小: {total_size} 字符")
    except:
        pass

if __name__ == "__main__":
    # 可选：通过命令行参数指定输出文件名
    output_filename = 'combined_code.txt'
    
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]
        if not output_filename.endswith('.txt'):
            output_filename += '.txt'
    
    combine_py_files(output_filename)