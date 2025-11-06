# remove_gender_symbols.py
"""
删除文件夹名中的性别符号（♀和♂）的脚本
支持批量处理、预览模式、递归处理等功能
"""

import os
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging

class FolderRenamer:
    """文件夹重命名工具"""
    
    def __init__(self, root_path, recursive=True, dry_run=False, backup=False):
        """
        初始化重命名工具
        
        Args:
            root_path: 根目录路径
            recursive: 是否递归处理子目录
            dry_run: 是否为预览模式（不实际执行）
            backup: 是否创建备份
        """
        self.root_path = Path(root_path)
        self.recursive = recursive
        self.dry_run = dry_run
        self.backup = backup
        
        # 要删除的符号
        self.symbols_to_remove = ['♀', '♂']
        
        # 操作记录
        self.operations = []
        self.conflicts = []
        self.errors = []
        
        # 设置日志
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志记录"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('folder_rename.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def scan_folders(self):
        """扫描需要重命名的文件夹"""
        folders_to_rename = []
        
        if self.recursive:
            # 递归扫描所有子目录
            for root, dirs, files in os.walk(self.root_path):
                root_path = Path(root)
                for dir_name in dirs:
                    dir_path = root_path / dir_name
                    if self.needs_rename(dir_name):
                        new_name = self.clean_name(dir_name)
                        folders_to_rename.append({
                            'original_path': dir_path,
                            'original_name': dir_name,
                            'new_name': new_name,
                            'parent_dir': root_path
                        })
        else:
            # 只处理直接子目录
            if self.root_path.exists():
                for item in self.root_path.iterdir():
                    if item.is_dir():
                        dir_name = item.name
                        if self.needs_rename(dir_name):
                            new_name = self.clean_name(dir_name)
                            folders_to_rename.append({
                                'original_path': item,
                                'original_name': dir_name,
                                'new_name': new_name,
                                'parent_dir': self.root_path
                            })
        
        return folders_to_rename
    
    def needs_rename(self, folder_name):
        """检查文件夹名是否需要重命名"""
        return any(symbol in folder_name for symbol in self.symbols_to_remove)
    
    def clean_name(self, folder_name):
        """清理文件夹名，删除特殊符号"""
        cleaned_name = folder_name
        for symbol in self.symbols_to_remove:
            cleaned_name = cleaned_name.replace(symbol, '')
        
        # 清理多余的空格
        cleaned_name = ' '.join(cleaned_name.split())
        
        # 去除首尾空格
        cleaned_name = cleaned_name.strip()
        
        # 如果清理后为空，使用默认名称
        if not cleaned_name:
            cleaned_name = 'unnamed_folder'
            
        return cleaned_name
    
    def check_conflicts(self, folders_to_rename):
        """检查重命名冲突"""
        conflicts = []
        name_counts = {}
        
        for folder_info in folders_to_rename:
            parent_dir = folder_info['parent_dir']
            new_name = folder_info['new_name']
            new_path = parent_dir / new_name
            
            # 检查目标路径是否已存在
            if new_path.exists() and new_path != folder_info['original_path']:
                conflicts.append({
                    'original': folder_info['original_path'],
                    'target': new_path,
                    'reason': '目标路径已存在'
                })
            
            # 检查是否有重复的新名称
            key = str(parent_dir / new_name)
            if key in name_counts:
                name_counts[key].append(folder_info)
            else:
                name_counts[key] = [folder_info]
        
        # 处理重复名称
        for key, folder_list in name_counts.items():
            if len(folder_list) > 1:
                for i, folder_info in enumerate(folder_list):
                    if i > 0:  # 第一个保持原样，后续的添加后缀
                        conflicts.append({
                            'original': folder_info['original_path'],
                            'target': Path(key),
                            'reason': f'重复名称，建议添加后缀_{i}'
                        })
        
        return conflicts
    
    def resolve_conflicts(self, folders_to_rename):
        """解决命名冲突"""
        resolved_folders = []
        name_counts = {}
        
        for folder_info in folders_to_rename.copy():
            parent_dir = folder_info['parent_dir']
            new_name = folder_info['new_name']
            
            # 生成唯一名称
            unique_name = self.generate_unique_name(parent_dir, new_name, name_counts)
            
            folder_info['new_name'] = unique_name
            folder_info['new_path'] = parent_dir / unique_name
            
            resolved_folders.append(folder_info)
            
            # 记录名称使用情况
            key = str(parent_dir)
            if key not in name_counts:
                name_counts[key] = set()
            name_counts[key].add(unique_name)
        
        return resolved_folders
    
    def generate_unique_name(self, parent_dir, base_name, name_counts):
        """生成唯一的文件夹名称"""
        key = str(parent_dir)
        if key not in name_counts:
            name_counts[key] = set()
        
        # 检查基础名称是否可用
        if base_name not in name_counts[key] and not (parent_dir / base_name).exists():
            return base_name
        
        # 生成带后缀的名称
        counter = 1
        while True:
            candidate_name = f"{base_name}_{counter}"
            if (candidate_name not in name_counts[key] and 
                not (parent_dir / candidate_name).exists()):
                return candidate_name
            counter += 1
    
    def create_backup(self, folder_path):
        """创建文件夹备份"""
        if not self.backup:
            return None
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{folder_path.name}_backup_{timestamp}"
            backup_path = folder_path.parent / backup_name
            
            shutil.copytree(folder_path, backup_path)
            self.logger.info(f"创建备份: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"创建备份失败 {folder_path}: {e}")
            return None
    
    def rename_folder(self, folder_info):
        """重命名单个文件夹"""
        original_path = folder_info['original_path']
        new_path = folder_info['new_path']
        
        try:
            # 创建备份（如果需要）
            backup_path = None
            if self.backup:
                backup_path = self.create_backup(original_path)
            
            # 执行重命名
            if not self.dry_run:
                original_path.rename(new_path)
            
            # 记录操作
            operation = {
                'original_path': str(original_path),
                'new_path': str(new_path),
                'original_name': folder_info['original_name'],
                'new_name': folder_info['new_name'],
                'backup_path': str(backup_path) if backup_path else None,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            self.operations.append(operation)
            
            if self.dry_run:
                self.logger.info(f"[预览] 重命名: {original_path} -> {new_path}")
            else:
                self.logger.info(f"重命名成功: {original_path} -> {new_path}")
                
            return True
            
        except Exception as e:
            error_info = {
                'original_path': str(original_path),
                'new_path': str(new_path),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self.errors.append(error_info)
            self.logger.error(f"重命名失败 {original_path}: {e}")
            return False
    
    def process_folders(self):
        """处理所有需要重命名的文件夹"""
        if not self.root_path.exists():
            self.logger.error(f"根目录不存在: {self.root_path}")
            return False
        
        self.logger.info(f"开始扫描目录: {self.root_path}")
        self.logger.info(f"递归模式: {self.recursive}")
        self.logger.info(f"预览模式: {self.dry_run}")
        self.logger.info(f"备份模式: {self.backup}")
        
        # 扫描需要重命名的文件夹
        folders_to_rename = self.scan_folders()
        
        if not folders_to_rename:
            self.logger.info("没有找到需要重命名的文件夹")
            return True
        
        self.logger.info(f"找到 {len(folders_to_rename)} 个需要重命名的文件夹")
        
        # 检查冲突
        conflicts = self.check_conflicts(folders_to_rename)
        if conflicts:
            self.logger.warning(f"发现 {len(conflicts)} 个潜在冲突")
            for conflict in conflicts:
                self.logger.warning(f"冲突: {conflict['original']} -> {conflict['target']} ({conflict['reason']})")
        
        # 解决冲突
        resolved_folders = self.resolve_conflicts(folders_to_rename)
        
        # 显示预览
        self.show_preview(resolved_folders)
        
        # 如果不是预览模式，询问用户确认
        if not self.dry_run:
            if not self.confirm_operation():
                self.logger.info("操作已取消")
                return False
        
        # 执行重命名
        success_count = 0
        for folder_info in resolved_folders:
            if self.rename_folder(folder_info):
                success_count += 1
        
        # 输出统计结果
        self.show_summary(success_count, len(resolved_folders))
        
        # 保存操作记录
        self.save_log()
        
        return len(self.errors) == 0
    
    def show_preview(self, folders_to_rename):
        """显示重命名预览"""
        print("\n" + "="*80)
        print("重命名预览")
        print("="*80)
        
        for i, folder_info in enumerate(folders_to_rename, 1):
            print(f"{i:3d}. {folder_info['original_name']}")
            print(f"     -> {folder_info['new_name']}")
            print(f"     路径: {folder_info['original_path']}")
            print()
        
        print(f"总计: {len(folders_to_rename)} 个文件夹")
        print("="*80)
    
    def confirm_operation(self):
        """确认是否执行操作"""
        while True:
            response = input("\n是否确认执行重命名操作？(y/n): ").lower().strip()
            if response in ['y', 'yes', '是']:
                return True
            elif response in ['n', 'no', '否']:
                return False
            else:
                print("请输入 y/yes/是 或 n/no/否")
    
    def show_summary(self, success_count, total_count):
        """显示操作总结"""
        print("\n" + "="*50)
        print("操作总结")
        print("="*50)
        print(f"成功: {success_count}/{total_count}")
        print(f"失败: {len(self.errors)}")
        print(f"冲突: {len(self.conflicts)}")
        
        if self.errors:
            print("\n失败的操作:")
            for error in self.errors:
                print(f"  - {error['original_path']}: {error['error']}")
        
        print("="*50)
    
    def save_log(self):
        """保存操作日志"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'root_path': str(self.root_path),
            'recursive': self.recursive,
            'dry_run': self.dry_run,
            'backup': self.backup,
            'symbols_removed': self.symbols_to_remove,
            'operations': self.operations,
            'conflicts': self.conflicts,
            'errors': self.errors,
            'summary': {
                'total_operations': len(self.operations),
                'successful_operations': len([op for op in self.operations if op['status'] == 'success']),
                'failed_operations': len(self.errors),
                'conflicts_found': len(self.conflicts)
            }
        }
        
        log_filename = f"rename_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(log_filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"操作日志已保存: {log_filename}")
            
        except Exception as e:
            self.logger.error(f"保存日志失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='删除文件夹名中的性别符号（♀和♂）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 预览模式，查看会重命名哪些文件夹
  python remove_gender_symbols.py /path/to/folders --dry-run
  
  # 实际执行重命名（只处理直接子目录）
  python remove_gender_symbols.py /path/to/folders --no-recursive
  
  # 递归处理所有子目录并创建备份
  python remove_gender_symbols.py /path/to/folders --backup
  
  # 交互模式，手动确认每个操作
  python remove_gender_symbols.py /path/to/folders --interactive
        """
    )
    
    parser.add_argument('path', type=str, help='要处理的根目录路径')
    
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='递归处理所有子目录（默认启用）')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive',
                       help='只处理直接子目录')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式，不实际执行重命名')
    
    parser.add_argument('--backup', action='store_true',
                       help='重命名前创建文件夹备份')
    
    parser.add_argument('--interactive', action='store_true',
                       help='交互模式，每次操作前确认')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='设置日志级别')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 验证路径
    if not os.path.exists(args.path):
        print(f"错误: 路径不存在 - {args.path}")
        return 1
    
    if not os.path.isdir(args.path):
        print(f"错误: 路径不是目录 - {args.path}")
        return 1
    
    # 创建重命名工具
    renamer = FolderRenamer(
        root_path=args.path,
        recursive=args.recursive,
        dry_run=args.dry_run,
        backup=args.backup
    )
    
    try:
        # 处理文件夹
        success = renamer.process_folders()
        
        if success:
            print("\n✅ 操作完成!")
            return 0
        else:
            print("\n❌ 操作过程中出现错误，请查看日志")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
