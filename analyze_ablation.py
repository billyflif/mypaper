#!/usr/bin/env python3
# analyze_ablation_results.py - 消融实验结果分析和可视化

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description='分析消融实验结果')
    parser.add_argument('--results-dir', type=str, default='ablation_results',
                       help='消融实验结果目录')
    parser.add_argument('--output-dir', type=str, default='ablation_analysis',
                       help='分析结果输出目录')
    parser.add_argument('--format', type=str, choices=['png', 'pdf', 'svg'], 
                       default='png', help='图片保存格式')
    return parser.parse_args()

class AblationAnalyzer:
    """消融实验结果分析器"""
    
    def __init__(self, results_dir, output_dir, img_format='png'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.img_format = img_format
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        self.data = self._load_experiment_data()
        
    def _load_experiment_data(self):
        """加载所有实验数据"""
        data = []
        
        # 查找所有实验目录
        exp_dirs = [d for d in os.listdir(self.results_dir) 
                   if os.path.isdir(os.path.join(self.results_dir, d))]
        
        for exp_dir in exp_dirs:
            exp_path = os.path.join(self.results_dir, exp_dir)
            result_file = os.path.join(exp_path, 'experiment_result.json')
            
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    exp_data = json.load(f)
                    data.append(exp_data)
            else:
                print(f"警告: 找不到实验结果文件 {result_file}")
        
        return data
    
    def generate_comprehensive_analysis(self):
        """生成综合分析报告"""
        print("开始生成综合分析...")
        
        # 1. 基本统计分析
        self._generate_basic_statistics()
        
        # 2. 性能对比分析
        self._generate_performance_comparison()
        
        # 3. 组件重要性分析
        self._analyze_component_importance()
        
        # 4. 参数效率分析
        self._analyze_parameter_efficiency()
        
        # 5. 损失函数贡献分析
        self._analyze_loss_contributions()
        
        # 6. 生成详细报告
        self._generate_detailed_report()
        
        print(f"分析完成！结果保存在: {self.output_dir}")
    
    def _generate_basic_statistics(self):
        """生成基本统计信息"""
        if not self.data:
            print("没有找到实验数据")
            return
        
        # 创建DataFrame
        df_data = []
        for exp in self.data:
            df_data.append({
                'experiment': exp['exp_name'],
                'description': exp['description'],
                'best_val_acc': exp['best_val_acc'],
                'total_params': exp.get('total_params', 0),
                'trainable_params': exp.get('trainable_params', 0),
                'disabled_components': len(exp.get('disabled_components', []))
            })
        
        df = pd.DataFrame(df_data)
        
        # 保存统计信息
        stats_file = os.path.join(self.output_dir, 'basic_statistics.csv')
        df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        
        # 生成统计图
        self._plot_basic_statistics(df)
        
        print(f"基本统计信息已保存到: {stats_file}")
    
    def _plot_basic_statistics(self, df):
        """绘制基本统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 准确率分布
        axes[0, 0].hist(df['best_val_acc'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('验证准确率分布')
        axes[0, 0].set_xlabel('准确率')
        axes[0, 0].set_ylabel('实验数量')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 参数量分布
        if df['total_params'].sum() > 0:
            axes[0, 1].scatter(df['total_params']/1e6, df['best_val_acc'], alpha=0.7, s=60)
            axes[0, 1].set_title('参数量 vs 准确率')
            axes[0, 1].set_xlabel('参数量 (M)')
            axes[0, 1].set_ylabel('验证准确率')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 禁用组件数量 vs 性能
        axes[1, 0].scatter(df['disabled_components'], df['best_val_acc'], alpha=0.7, s=60, c='orange')
        axes[1, 0].set_title('禁用组件数量 vs 性能')
        axes[1, 0].set_xlabel('禁用组件数量')
        axes[1, 0].set_ylabel('验证准确率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Top性能实验
        top_df = df.nlargest(5, 'best_val_acc')
        axes[1, 1].barh(range(len(top_df)), top_df['best_val_acc'], alpha=0.7, color='lightgreen')
        axes[1, 1].set_yticks(range(len(top_df)))
        axes[1, 1].set_yticklabels([exp[:15] + '...' if len(exp) > 15 else exp 
                                   for exp in top_df['experiment']], fontsize=9)
        axes[1, 1].set_title('Top 5 性能实验')
        axes[1, 1].set_xlabel('验证准确率')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'basic_statistics.{self.img_format}'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_performance_comparison(self):
        """生成性能对比分析"""
        # 创建性能对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 准备数据
        exp_names = [exp['exp_name'] for exp in self.data]
        accuracies = [exp['best_val_acc'] for exp in self.data]
        
        # 按准确率排序
        sorted_data = sorted(zip(exp_names, accuracies), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_accs = zip(*sorted_data)
        
        # 1. 水平条形图
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_names)))
        bars = ax1.barh(range(len(sorted_names)), sorted_accs, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(sorted_names)))
        ax1.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                            for name in sorted_names], fontsize=9)
        ax1.set_xlabel('验证准确率')
        ax1.set_title('消融实验性能排名')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, sorted_accs)):
            ax1.text(acc + 0.001, i, f'{acc:.4f}', va='center', fontsize=8)
        
        # 2. 相对基线的性能变化
        baseline_acc = None
        for exp in self.data:
            if exp['exp_name'] == 'baseline':
                baseline_acc = exp['best_val_acc']
                break
        
        if baseline_acc:
            relative_changes = [(acc - baseline_acc) / baseline_acc * 100 
                              for acc in accuracies]
            
            # 按变化幅度排序
            relative_data = sorted(zip(exp_names, relative_changes), 
                                 key=lambda x: x[1], reverse=True)
            rel_names, rel_changes = zip(*relative_data)
            
            colors = ['green' if x >= 0 else 'red' for x in rel_changes]
            bars2 = ax2.barh(range(len(rel_names)), rel_changes, color=colors, alpha=0.7)
            ax2.set_yticks(range(len(rel_names)))
            ax2.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                               for name in rel_names], fontsize=9)
            ax2.set_xlabel('相对基线的性能变化 (%)')
            ax2.set_title('相对基线性能变化')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # 添加数值标签
            for i, (bar, change) in enumerate(zip(bars2, rel_changes)):
                ax2.text(change + (0.5 if change >= 0 else -0.5), i, 
                        f'{change:+.2f}%', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'performance_comparison.{self.img_format}'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_component_importance(self):
        """分析组件重要性"""
        baseline_acc = None
        for exp in self.data:
            if exp['exp_name'] == 'baseline':
                baseline_acc = exp['best_val_acc']
                break
        
        if not baseline_acc:
            print("警告: 未找到基线实验，跳过组件重要性分析")
            return
        
        # 分析单组件禁用的影响
        component_impacts = {}
        
        for exp in self.data:
            disabled = exp.get('disabled_components', [])
            if len(disabled) == 1:  # 只禁用一个组件
                component = disabled[0]
                impact = baseline_acc - exp['best_val_acc']
                component_impacts[component] = {
                    'performance_drop': impact,
                    'relative_drop': impact / baseline_acc * 100,
                    'experiment': exp['exp_name']
                }
        
        if not component_impacts:
            print("警告: 未找到单组件消融实验")
            return
        
        # 生成组件重要性图
        components = list(component_impacts.keys())
        drops = [component_impacts[comp]['performance_drop'] for comp in components]
        relative_drops = [component_impacts[comp]['relative_drop'] for comp in components]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绝对性能下降
        colors = plt.cm.Reds(np.linspace(0.3, 1, len(components)))
        bars1 = ax1.bar(range(len(components)), drops, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(components)))
        ax1.set_xticklabels(components, rotation=45, ha='right')
        ax1.set_ylabel('性能下降 (绝对值)')
        ax1.set_title('组件重要性 - 绝对性能影响')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, drop in zip(bars1, drops):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{drop:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 相对性能下降
        bars2 = ax2.bar(range(len(components)), relative_drops, color=colors, alpha=0.8)
        ax2.set_xticks(range(len(components)))
        ax2.set_xticklabels(components, rotation=45, ha='right')
        ax2.set_ylabel('性能下降 (%)')
        ax2.set_title('组件重要性 - 相对性能影响')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, drop in zip(bars2, relative_drops):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{drop:.2f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'component_importance.{self.img_format}'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存组件重要性数据
        importance_df = pd.DataFrame([
            {
                'component': comp,
                'performance_drop': data['performance_drop'],
                'relative_drop_percent': data['relative_drop'],
                'experiment': data['experiment']
            }
            for comp, data in component_impacts.items()
        ])
        importance_df = importance_df.sort_values('performance_drop', ascending=False)
        importance_df.to_csv(os.path.join(self.output_dir, 'component_importance.csv'), 
                           index=False, encoding='utf-8-sig')
    
    def _analyze_parameter_efficiency(self):
        """分析参数效率"""
        # 收集参数和性能数据
        param_data = []
        for exp in self.data:
            if exp.get('total_params', 0) > 0:
                param_data.append({
                    'experiment': exp['exp_name'],
                    'accuracy': exp['best_val_acc'],
                    'total_params': exp['total_params'],
                    'trainable_params': exp['trainable_params'],
                    'efficiency': exp['best_val_acc'] / (exp['total_params'] / 1e6)  # 准确率/M参数
                })
        
        if not param_data:
            print("警告: 没有找到参数信息，跳过效率分析")
            return
        
        df = pd.DataFrame(param_data)
        
        # 生成效率分析图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 参数量 vs 准确率散点图
        scatter = ax1.scatter(df['total_params']/1e6, df['accuracy'], 
                            s=60, alpha=0.7, c=df['efficiency'], cmap='viridis')
        ax1.set_xlabel('参数量 (M)')
        ax1.set_ylabel('验证准确率')
        ax1.set_title('参数量 vs 性能')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='效率 (准确率/M参数)')
        
        # 2. 效率排名
        df_sorted = df.sort_values('efficiency', ascending=True)
        bars = ax2.barh(range(len(df_sorted)), df_sorted['efficiency'], alpha=0.7)
        ax2.set_yticks(range(len(df_sorted)))
        ax2.set_yticklabels([name[:15] + '...' if len(name) > 15 else name 
                           for name in df_sorted['experiment']], fontsize=9)
        ax2.set_xlabel('效率 (准确率/M参数)')
        ax2.set_title('参数效率排名')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. 可训练参数 vs 总参数
        ax3.scatter(df['total_params']/1e6, df['trainable_params']/1e6, 
                   s=60, alpha=0.7, c=df['accuracy'], cmap='plasma')
        ax3.plot([0, df['total_params'].max()/1e6], [0, df['total_params'].max()/1e6], 
                'r--', alpha=0.5, label='y=x')
        ax3.set_xlabel('总参数量 (M)')
        ax3.set_ylabel('可训练参数量 (M)')
        ax3.set_title('参数组成分析')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 参数减少 vs 性能损失
        baseline_params = None
        baseline_acc = None
        for exp in self.data:
            if exp['exp_name'] == 'baseline':
                baseline_params = exp.get('total_params', 0)
                baseline_acc = exp['best_val_acc']
                break
        
        if baseline_params and baseline_acc:
            param_reductions = []
            acc_losses = []
            exp_names = []
            
            for exp in self.data:
                if exp.get('total_params', 0) > 0 and exp['exp_name'] != 'baseline':
                    param_reduction = (baseline_params - exp['total_params']) / baseline_params * 100
                    acc_loss = baseline_acc - exp['best_val_acc']
                    param_reductions.append(param_reduction)
                    acc_losses.append(acc_loss)
                    exp_names.append(exp['exp_name'])
            
            scatter2 = ax4.scatter(param_reductions, acc_losses, s=60, alpha=0.7)
            ax4.set_xlabel('参数减少 (%)')
            ax4.set_ylabel('准确率损失')
            ax4.set_title('参数压缩 vs 性能损失')
            ax4.grid(True, alpha=0.3)
            
            # 添加标签
            for i, name in enumerate(exp_names):
                if i % 2 == 0:  # 只显示部分标签避免重叠
                    ax4.annotate(name[:10], (param_reductions[i], acc_losses[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'parameter_efficiency.{self.img_format}'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存效率数据
        df.to_csv(os.path.join(self.output_dir, 'parameter_efficiency.csv'), 
                 index=False, encoding='utf-8-sig')
    
    def _analyze_loss_contributions(self):
        """分析损失函数贡献"""
        loss_experiments = []
        for exp in self.data:
            exp_name = exp['exp_name']
            if any(loss in exp_name for loss in ['no_reconstruction', 'no_lsc', 'no_auxiliary', 
                                               'no_center', 'no_domain']):
                loss_experiments.append(exp)
        
        if not loss_experiments:
            print("警告: 未找到损失函数消融实验")
            return
        
        # 找到基线
        baseline_acc = None
        for exp in self.data:
            if exp['exp_name'] == 'baseline':
                baseline_acc = exp['best_val_acc']
                break
        
        if not baseline_acc:
            return
        
        # 分析各损失函数的贡献
        loss_impacts = {}
        loss_mapping = {
            'no_reconstruction': '重建损失',
            'no_lsc': '潜在一致性损失',
            'no_auxiliary': '辅助判别损失',
            'no_center': 'Center损失',
            'no_domain': '域适应损失'
        }
        
        for exp in loss_experiments:
            for key, chinese_name in loss_mapping.items():
                if key in exp['exp_name']:
                    impact = baseline_acc - exp['best_val_acc']
                    relative_impact = impact / baseline_acc * 100
                    loss_impacts[chinese_name] = {
                        'absolute_impact': impact,
                        'relative_impact': relative_impact,
                        'experiment': exp['exp_name']
                    }
                    break
        
        if not loss_impacts:
            return
        
        # 生成损失贡献图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        loss_names = list(loss_impacts.keys())
        absolute_impacts = [loss_impacts[name]['absolute_impact'] for name in loss_names]
        relative_impacts = [loss_impacts[name]['relative_impact'] for name in loss_names]
        
        # 绝对影响
        colors = plt.cm.Blues(np.linspace(0.4, 1, len(loss_names)))
        bars1 = ax1.bar(range(len(loss_names)), absolute_impacts, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(loss_names)))
        ax1.set_xticklabels(loss_names, rotation=45, ha='right')
        ax1.set_ylabel('性能下降 (绝对值)')
        ax1.set_title('损失函数贡献 - 绝对影响')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, impact in zip(bars1, absolute_impacts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{impact:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 相对影响
        bars2 = ax2.bar(range(len(loss_names)), relative_impacts, color=colors, alpha=0.8)
        ax2.set_xticks(range(len(loss_names)))
        ax2.set_xticklabels(loss_names, rotation=45, ha='right')
        ax2.set_ylabel('性能下降 (%)')
        ax2.set_title('损失函数贡献 - 相对影响')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, impact in zip(bars2, relative_impacts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{impact:.2f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'loss_contributions.{self.img_format}'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存损失贡献数据
        loss_df = pd.DataFrame([
            {
                'loss_function': name,
                'absolute_impact': data['absolute_impact'],
                'relative_impact_percent': data['relative_impact'],
                'experiment': data['experiment']
            }
            for name, data in loss_impacts.items()
        ])
        loss_df = loss_df.sort_values('absolute_impact', ascending=False)
        loss_df.to_csv(os.path.join(self.output_dir, 'loss_contributions.csv'), 
                      index=False, encoding='utf-8-sig')
    
    def _generate_detailed_report(self):
        """生成详细的分析报告"""
        report_path = os.path.join(self.output_dir, 'detailed_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# SGPD-Net 消融实验详细分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. 实验概览
            f.write("## 1. 实验概览\n\n")
            f.write(f"- 总实验数量: {len(self.data)}\n")
            
            if self.data:
                accuracies = [exp['best_val_acc'] for exp in self.data]
                f.write(f"- 最高准确率: {max(accuracies):.4f}\n")
                f.write(f"- 最低准确率: {min(accuracies):.4f}\n")
                f.write(f"- 平均准确率: {np.mean(accuracies):.4f}\n")
                f.write(f"- 准确率标准差: {np.std(accuracies):.4f}\n\n")
            
            # 2. 关键发现
            f.write("## 2. 关键发现\n\n")
            
            # 找到最佳和最差实验
            if self.data:
                best_exp = max(self.data, key=lambda x: x['best_val_acc'])
                worst_exp = min(self.data, key=lambda x: x['best_val_acc'])
                
                f.write(f"### 最佳性能实验\n")
                f.write(f"- 实验名称: {best_exp['exp_name']}\n")
                f.write(f"- 描述: {best_exp['description']}\n")
                f.write(f"- 验证准确率: {best_exp['best_val_acc']:.4f}\n\n")
                
                f.write(f"### 最差性能实验\n")
                f.write(f"- 实验名称: {worst_exp['exp_name']}\n")
                f.write(f"- 描述: {worst_exp['description']}\n")
                f.write(f"- 验证准确率: {worst_exp['best_val_acc']:.4f}\n\n")
            
            # 3. 组件重要性排序
            baseline_acc = None
            for exp in self.data:
                if exp['exp_name'] == 'baseline':
                    baseline_acc = exp['best_val_acc']
                    break
            
            if baseline_acc:
                f.write("## 3. 组件重要性排序\n\n")
                f.write("基于单组件消融实验的性能下降程度排序:\n\n")
                
                component_impacts = []
                for exp in self.data:
                    disabled = exp.get('disabled_components', [])
                    if len(disabled) == 1:
                        component = disabled[0]
                        impact = baseline_acc - exp['best_val_acc']
                        component_impacts.append((component, impact, exp['exp_name']))
                
                component_impacts.sort(key=lambda x: x[1], reverse=True)
                
                for i, (component, impact, exp_name) in enumerate(component_impacts, 1):
                    f.write(f"{i}. **{component}**: {impact:.4f} ({impact/baseline_acc*100:.2f}%) - {exp_name}\n")
                
                f.write("\n")
            
            # 4. 建议和结论
            f.write("## 4. 建议和结论\n\n")
            f.write("### 架构设计建议\n")
            f.write("- 基于实验结果，以下组件对性能最为关键:\n")
            f.write("  - [根据实际结果填写最重要的组件]\n")
            f.write("  - [第二重要的组件]\n")
            f.write("  - [第三重要的组件]\n\n")
            
            f.write("### 训练策略建议\n")
            f.write("- 损失函数权重建议:\n")
            f.write("  - [根据损失函数消融结果给出建议]\n\n")
            
            f.write("### 计算效率建议\n")
            f.write("- 在资源受限的场景下，可以考虑:\n")
            f.write("  - [根据参数效率分析给出建议]\n\n")
            
            # 5. 附录 - 所有实验结果
            f.write("## 5. 附录 - 所有实验结果\n\n")
            f.write("| 实验名称 | 描述 | 验证准确率 | 禁用组件 |\n")
            f.write("|----------|------|------------|----------|\n")
            
            sorted_exps = sorted(self.data, key=lambda x: x['best_val_acc'], reverse=True)
            for exp in sorted_exps:
                disabled = ', '.join(exp.get('disabled_components', ['无']))
                f.write(f"| {exp['exp_name']} | {exp['description'][:30]}... | {exp['best_val_acc']:.4f} | {disabled} |\n")
        
        print(f"详细分析报告已保存到: {report_path}")

def main():
    args = parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"错误: 结果目录不存在 - {args.results_dir}")
        return
    
    # 创建分析器
    analyzer = AblationAnalyzer(args.results_dir, args.output_dir, args.format)
    
    # 生成综合分析
    analyzer.generate_comprehensive_analysis()

if __name__ == "__main__":
    main()