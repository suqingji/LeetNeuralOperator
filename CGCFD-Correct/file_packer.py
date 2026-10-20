#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件打包/解包工具
================

功能：
1. 压缩模式：将指定目录下的代码文件和文档合并成.dat文件
2. 解压模式：从.dat文件还原原始文件结构

用法：
    # 压缩模式
    python file_packer.py compress /path/to/source -o /path/to/output

    # 解压模式
    python file_packer.py decompress /path/to/packed.dat -d /path/to/restore

作者: H2 Assessment Team
日期: 2026-09-23
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Set
import re


# 文件分隔符（使用特殊标记，避免与正常内容冲突）
SEPARATOR = "=" * 80
FILE_HEADER_START = "<<<<FILE_START>>>>"
FILE_HEADER_END = "<<<<FILE_END>>>>"

# 默认支持的文件扩展名
DEFAULT_EXTENSIONS = {
    '.py',      # Python源代码
    '.md',      # Markdown文档
    '.txt',     # 纯文本
    '.json',    # JSON配置
    '.yaml',    # YAML配置
    '.yml',     # YAML配置
    '.ini',     # INI配置
    '.cfg',     # 配置文件
    '.html',    # HTML
    '.css',     # CSS样式
    '.js',      # JavaScript
    '.ts',      # TypeScript
    '.java',    # Java
    '.c',       # C语言
    '.cpp',     # C++
    '.h',       # C/C++头文件
    '.sh',      # Shell脚本
    '.bat',     # Windows批处理
    '.sql',     # SQL脚本
    '.xml',     # XML
}


class FilePacker:
    """文件打包器"""

    def __init__(self, max_lines_per_file: int = 3000):
        self.max_lines_per_file = max_lines_per_file
        self.current_file_index = 1
        self.current_line_count = 0
        self.current_output_file = None
        self.output_path = None   # <-- 新增：记录输出目录

    def compress(self, source_dir: str, output_dir: str, extensions: Set[str] = None):
        """
        压缩模式：将目录下的文件打包成.dat文件

        Args:
            source_dir: 源目录路径
            output_dir: 输出目录路径
            extensions: 要打包的文件扩展名集合
        """
        if extensions is None:
            extensions = DEFAULT_EXTENSIONS

        source_path = Path(source_dir)
        output_path = Path(output_dir)

        if not source_path.exists():
            print(f"错误: 源目录不存在: {source_dir}")
            return False

        if not source_path.is_dir():
            print(f"错误: 源路径不是目录: {source_dir}")
            return False

        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存输出路径到实例变量（供后续创建新文件时使用）
        self.output_path = output_path

        # 初始化第一个输出文件
        self._open_new_output_file()

        # 收集所有目标文件
        files_to_pack = []
        for root, dirs, files in os.walk(source_path):
            # 跳过隐藏目录和常见非代码目录
            dirs[:] = [d for d in dirs if not d.startswith('.')
                      and d not in {'__pycache__', 'node_modules', '.git', 'venv', 'env'}]

            for filename in files:
                file_path = Path(root) / filename
                if file_path.suffix.lower() in extensions:
                    files_to_pack.append(file_path)

        if not files_to_pack:
            print("警告: 未找到符合条件的文件")
            return False

        print(f"找到 {len(files_to_pack)} 个文件待打包")

        # 逐个打包文件
        packed_count = 0
        for file_path in sorted(files_to_pack):
            try:
                self._pack_single_file(file_path, source_path)
                packed_count += 1
                print(f"  ✓ {file_path.relative_to(source_path)}")
            except Exception as e:
                print(f"  ✗ {file_path.relative_to(source_path)}: {e}")

        # 关闭最后一个文件
        if self.current_output_file:
            self.current_output_file.close()

        print(f"\n打包完成!")
        print(f"  已处理: {packed_count}/{len(files_to_pack)} 个文件")
        print(f"  输出目录: {output_path.absolute()}")
        print(f"  生成文件: {self.current_file_index} 个.dat文件")

        return True

    def _open_new_output_file(self):
        """打开新的输出文件"""
        if self.current_output_file:
            self.current_output_file.close()

        filename = f"{self.current_file_index}.dat"
        filepath = self.output_path / filename
        self.current_output_file = open(filepath, 'w', encoding='utf-8')
        self.current_line_count = 0

        print(f"创建输出文件: {filename}")

    def _pack_single_file(self, file_path: Path, base_path: Path):
        """打包单个文件"""
        # 计算相对路径
        rel_path = file_path.relative_to(base_path)

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.splitlines(keepends=True)
        line_count = len(lines)

        # 检查是否需要切换到新文件
        # 预留空间：文件头约10行 + 分隔符5行 + 实际内容
        estimated_lines = line_count + 15

        if self.current_line_count + estimated_lines > self.max_lines_per_file:
            self.current_file_index += 1
            self._open_new_output_file()  # <-- 不再传递参数，使用实例变量self.output_path

        # 写入文件头
        header = f"""{SEPARATOR}
{FILE_HEADER_START}
PATH: {rel_path}
LINES: {line_count}
ENCODING: utf-8
{FILE_HEADER_END}
{SEPARATOR}
"""
        self.current_output_file.write(header)
        self.current_line_count += header.count('\n')

        # 写入文件内容
        self.current_output_file.write(content)
        self.current_line_count += line_count

        # 如果内容不以换行结尾，添加一个
        if content and not content.endswith('\n'):
            self.current_output_file.write('\n')
            self.current_line_count += 1

        # 写入文件尾
        footer = f"\n{SEPARATOR}\n\n"
        self.current_output_file.write(footer)
        self.current_line_count += footer.count('\n')

    def decompress(self, input_path: str, output_dir: str):
        """
        解压模式：从.dat文件还原原始文件

        Args:
            input_path: 输入的.dat文件或目录
            output_dir: 输出目录路径
        """
        input_path_obj = Path(input_path)
        output_path = Path(output_dir)

        # 收集所有.dat文件
        dat_files = []
        if input_path_obj.is_file():
            if input_path_obj.suffix == '.dat':
                dat_files.append(input_path_obj)
            else:
                print(f"错误: 不是.dat文件: {input_path}")
                return False
        elif input_path_obj.is_dir():
            dat_files = sorted(input_path_obj.glob('*.dat'))
            if not dat_files:
                print(f"错误: 目录中没有.dat文件: {input_path}")
                return False
        else:
            print(f"错误: 路径不存在: {input_path}")
            return False

        print(f"找到 {len(dat_files)} 个.dat文件待解压")

        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)

        # 逐个处理.dat文件
        restored_count = 0
        for dat_file in dat_files:
            count = self._decompress_single_file(dat_file, output_path)
            restored_count += count

        print(f"\n解压完成!")
        print(f"  已恢复: {restored_count} 个文件")
        print(f"  输出目录: {output_path.absolute()}")

        return True

    def _decompress_single_file(self, dat_file: Path, output_path: Path) -> int:
        """解压单个.dat文件"""
        print(f"处理: {dat_file.name}")

        with open(dat_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用正则表达式分割文件块
        pattern = rf'{SEPARATOR}\n{FILE_HEADER_START}\n(.*?)\n{FILE_HEADER_END}\n{SEPARATOR}\n(.*?)(?={SEPARATOR}\n|$)'
        matches = re.findall(pattern, content, re.DOTALL)

        restored_count = 0
        for header_info, file_content in matches:
            try:
                # 解析文件头信息
                path_match = re.search(r'PATH:\s*(.+)', header_info)
                if not path_match:
                    print(f"  ✗ 无法解析文件路径")
                    continue

                rel_path = path_match.group(1).strip()

                # 构建完整输出路径
                output_file = output_path / rel_path

                # 创建父目录
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # 写入文件内容
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(file_content)

                restored_count += 1
                print(f"  ✓ {rel_path}")

            except Exception as e:
                print(f"  ✗ 恢复文件失败: {e}")

        return restored_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='文件打包/解包工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 压缩模式
  python file_packer.py compress ./my_project -o ./packed

  # 解压模式
  python file_packer.py decompress ./packed/1.dat -d ./restored

  # 解压整个目录
  python file_packer.py decompress ./packed -d ./restored
        """
    )

    subparsers = parser.add_subparsers(dest='mode', help='操作模式')

    # 压缩子命令
    compress_parser = subparsers.add_parser('compress', help='压缩模式：将文件打包成.dat')
    compress_parser.add_argument('source', help='源目录路径')
    compress_parser.add_argument('-o', '--output', default='./packed',
                                help='输出目录路径 (默认: ./packed)')
    compress_parser.add_argument('--max-lines', type=int, default=3000,
                                help='每个.dat文件的最大行数 (默认: 3000)')

    # 解压子命令
    decompress_parser = subparsers.add_parser('decompress', help='解压模式：从.dat还原文件')
    decompress_parser.add_argument('input', help='输入的.dat文件或目录')
    decompress_parser.add_argument('-d', '--dest', default='./restored',
                                  help='输出目录路径 (默认: ./restored)')

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return 1

    packer = FilePacker(max_lines_per_file=args.max_lines if hasattr(args, 'max_lines') else 3000)

    if args.mode == 'compress':
        success = packer.compress(args.source, args.output)
    elif args.mode == 'decompress':
        success = packer.decompress(args.input, args.dest)
    else:
        parser.print_help()
        return 1

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())