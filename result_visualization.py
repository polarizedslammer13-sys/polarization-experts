#!/usr/bin/env python3
"""
Pattern Resolution Ablation Study - Results Visualization
生成论文级别的可视化图表
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import pandas as pd

# 设置中文字体和论文风格
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10
rcParams['figure.dpi'] = 300
plt.style.use('seaborn-v0_8-paper')

def load_all_results(base_dir):
    """加载所有实验结果"""
    results = []
    
    # 查找所有report.json文件
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'report.json':
                with open(os.path.join(root, file), 'r') as f:
                    result = json.load(f)
                    results.append(result)
    
    # 按实验名称排序
    results.sort(key=lambda x: x['experiment'])
    
    return results


def create_matrix_data(results):
    """创建2D矩阵数据"""
    matrix_pcc = np.zeros((3, 3))
    matrix_ssim = np.zeros((3, 3))
    matrix_time = np.zeros((3, 3))
    
    # 映射关系
    speckle_map = {64: 0, 128: 1, 256: 2}
    model_map = {64: 0, 128: 1, 256: 2}
    
    for result in results:
        config = result['config']
        s_idx = speckle_map[config['speckle_size']]
        m_idx = model_map[config['model_output']]
        
        matrix_pcc[m_idx, s_idx] = result['test_pcc']
        matrix_ssim[m_idx, s_idx] = result['test_ssim']
        matrix_time[m_idx, s_idx] = result['time_hours']
    
    return matrix_pcc, matrix_ssim, matrix_time


def plot_figure1_model_output_effect(results, output_dir):
    """
    Figure 1: 固定散斑256，变化模型输出
    证明"高分辨率学习"的有效性
    """
    # 提取S256的实验
    s256_results = [r for r in results if r['config']['speckle_size'] == 256]
    s256_results.sort(key=lambda x: x['config']['model_output'])
    
    model_sizes = [r['config']['model_output'] for r in s256_results]
    pccs = [r['test_pcc'] for r in s256_results]
    ssims = [r['test_ssim'] for r in s256_results]
    times = [r['time_hours'] for r in s256_results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # (a) PCC vs Model Output
    ax = axes[0]
    ax.plot(model_sizes, pccs, 'o-', linewidth=2.5, markersize=8, 
           color='#2E86AB', label='PCC')
    ax.set_xlabel('Model Output Resolution', fontsize=11, fontweight='bold')
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=11, fontweight='bold')
    ax.set_title('(a) Reconstruction Accuracy vs Model Resolution', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(model_sizes)
    ax.set_xticklabels([f'{s}×{s}' for s in model_sizes])
    
    # 标注数值
    for x, y in zip(model_sizes, pccs):
        ax.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 8),
                   textcoords='offset points', ha='center', fontsize=9)
    
    # (b) SSIM vs Model Output
    ax = axes[1]
    ax.plot(model_sizes, ssims, 's-', linewidth=2.5, markersize=8,
           color='#A23B72', label='SSIM')
    ax.set_xlabel('Model Output Resolution', fontsize=11, fontweight='bold')
    ax.set_ylabel('Structural Similarity Index', fontsize=11, fontweight='bold')
    ax.set_title('(b) Structural Similarity vs Model Resolution',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(model_sizes)
    ax.set_xticklabels([f'{s}×{s}' for s in model_sizes])
    
    for x, y in zip(model_sizes, ssims):
        ax.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 8),
                   textcoords='offset points', ha='center', fontsize=9)
    
    # (c) Training Time
    ax = axes[2]
    bars = ax.bar(range(len(model_sizes)), times, color=['#06A77D', '#F18F01', '#C73E1D'])
    ax.set_xlabel('Model Output Resolution', fontsize=11, fontweight='bold')
    ax.set_ylabel('Training Time (hours)', fontsize=11, fontweight='bold')
    ax.set_title('(c) Computational Cost',
                fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(model_sizes)))
    ax.set_xticklabels([f'{s}×{s}' for s in model_sizes])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    for i, (bar, t) in enumerate(zip(bars, times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{t:.2f}h', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure1_ModelOutput_Effect.pdf'),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure1_ModelOutput_Effect.png'),
               dpi=300, bbox_inches='tight')
    print("✓ Figure 1 saved")


def plot_figure2_speckle_resolution_effect(results, output_dir):
    """
    Figure 2: 固定模型256，变化散斑分辨率
    揭示信息容量上限
    """
    # 提取M256的实验
    m256_results = [r for r in results if r['config']['model_output'] == 256]
    m256_results.sort(key=lambda x: x['config']['speckle_size'])
    
    speckle_sizes = [r['config']['speckle_size'] for r in m256_results]
    pccs = [r['test_pcc'] for r in m256_results]
    ssims = [r['test_ssim'] for r in m256_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) PCC & SSIM vs Speckle Resolution
    ax = axes[0]
    ax2 = ax.twinx()
    
    line1 = ax.plot(speckle_sizes, pccs, 'o-', linewidth=2.5, markersize=10,
                   color='#2E86AB', label='PCC')
    line2 = ax2.plot(speckle_sizes, ssims, 's-', linewidth=2.5, markersize=10,
                    color='#A23B72', label='SSIM')
    
    ax.set_xlabel('Speckle Resolution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pearson Correlation (PCC)', fontsize=11, fontweight='bold', color='#2E86AB')
    ax2.set_ylabel('Structural Similarity (SSIM)', fontsize=11, fontweight='bold', color='#A23B72')
    ax.set_title('(a) Information Capacity vs Speckle Resolution',
                fontsize=13, fontweight='bold')
    
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    
    ax.set_xticks(speckle_sizes)
    ax.set_xticklabels([f'{s}×{s}' for s in speckle_sizes])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 标注数值
    for x, y in zip(speckle_sizes, pccs):
        ax.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9, color='#2E86AB')
    
    for x, y in zip(speckle_sizes, ssims):
        ax2.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, -15),
                    textcoords='offset points', ha='center', fontsize=9, color='#A23B72')
    
    # (b) 信息损失分析
    ax = axes[1]
    
    # 计算相对于256×256的损失
    baseline_pcc = pccs[-1]
    pcc_loss = [(baseline_pcc - p) / baseline_pcc * 100 for p in pccs]
    
    bars = ax.bar(range(len(speckle_sizes)), pcc_loss, 
                 color=['#C73E1D', '#F18F01', '#06A77D'])
    ax.set_xlabel('Speckle Resolution', fontsize=12, fontweight='bold')
    ax.set_ylabel('PCC Loss Relative to 256×256 (%)', fontsize=11, fontweight='bold')
    ax.set_title('(b) Information Loss Analysis',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(speckle_sizes)))
    ax.set_xticklabels([f'{s}×{s}' for s in speckle_sizes])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    for i, (bar, loss) in enumerate(zip(bars, pcc_loss)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{loss:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure2_Speckle_Effect.pdf'),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure2_Speckle_Effect.png'),
               dpi=300, bbox_inches='tight')
    print("✓ Figure 2 saved")


def plot_figure3_efficiency_quality_tradeoff(results, output_dir):
    """
    Figure 3: 对角线配置的效率-质量权衡
    """
    # 提取对角线配置
    diagonal_configs = [(64, 64), (128, 128), (256, 256)]
    diagonal_results = []
    
    for s, m in diagonal_configs:
        for r in results:
            if r['config']['speckle_size'] == s and r['config']['model_output'] == m:
                diagonal_results.append(r)
                break
    
    configs = [f"{r['config']['speckle_size']}×{r['config']['model_output']}" 
              for r in diagonal_results]
    pccs = [r['test_pcc'] for r in diagonal_results]
    ssims = [r['test_ssim'] for r in diagonal_results]
    times = [r['time_hours'] for r in diagonal_results]
    params = [r['parameters_M'] for r in diagonal_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Scatter: Time vs PCC
    ax = axes[0]
    scatter = ax.scatter(times, pccs, s=[p*50 for p in params], 
                        c=range(len(configs)), cmap='viridis', 
                        alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for i, (t, p, cfg) in enumerate(zip(times, pccs, configs)):
        ax.annotate(cfg, xy=(t, p), xytext=(10, 5),
                   textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Training Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test PCC', fontsize=12, fontweight='bold')
    ax.set_title('(a) Quality-Efficiency Trade-off',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例说明气泡大小
    for param in [10, 30, 50]:
        ax.scatter([], [], s=param*50, c='gray', alpha=0.5, 
                  label=f'{param}M params')
    ax.legend(scatterpoints=1, frameon=True, labelspacing=1, title='Model Size')
    
    # (b) 性能总结表格
    ax = axes[1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Config', 'PCC', 'SSIM', 'Time(h)', 'Params(M)']
    
    for r in diagonal_results:
        cfg = f"{r['config']['speckle_size']}×{r['config']['model_output']}"
        row = [
            cfg,
            f"{r['test_pcc']:.4f}",
            f"{r['test_ssim']:.4f}",
            f"{r['time_hours']:.2f}",
            f"{r['parameters_M']:.1f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 交替行颜色
    for i in range(1, len(table_data)+1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')
    
    ax.set_title('(b) Performance Summary',
                fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure3_Efficiency_Tradeoff.pdf'),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure3_Efficiency_Tradeoff.png'),
               dpi=300, bbox_inches='tight')
    print("✓ Figure 3 saved")


def plot_figure4_heatmap(results, output_dir):
    """
    Figure 4: 2D热力图 - 全局视角
    """
    matrix_pcc, matrix_ssim, matrix_time = create_matrix_data(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    labels = ['64', '128', '256']
    
    # (a) PCC Heatmap
    ax = axes[0]
    im1 = ax.imshow(matrix_pcc, cmap='YlGnBu', aspect='auto', vmin=0.65, vmax=0.95)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Speckle Resolution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Output Resolution', fontsize=12, fontweight='bold')
    ax.set_title('(a) Test PCC', fontsize=13, fontweight='bold')
    
    # 标注数值
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{matrix_pcc[i, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    
    # (b) SSIM Heatmap
    ax = axes[1]
    im2 = ax.imshow(matrix_ssim, cmap='RdYlGn', aspect='auto', vmin=0.40, vmax=0.70)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Speckle Resolution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Output Resolution', fontsize=12, fontweight='bold')
    ax.set_title('(b) Test SSIM', fontsize=13, fontweight='bold')
    
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{matrix_ssim[i, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    
    # (c) Training Time Heatmap
    ax = axes[2]
    im3 = ax.imshow(matrix_time, cmap='Oranges', aspect='auto')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Speckle Resolution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Output Resolution', fontsize=12, fontweight='bold')
    ax.set_title('(c) Training Time (hours)', fontsize=13, fontweight='bold')
    
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{matrix_time[i, j]:.1f}',
                         ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure4_Heatmap.pdf'),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure4_Heatmap.png'),
               dpi=300, bbox_inches='tight')
    print("✓ Figure 4 saved")


def generate_summary_report(results, output_dir):
    """生成文字总结报告"""
    report = []
    report.append("="*80)
    report.append("PATTERN RESOLUTION ABLATION STUDY - SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    
    # 最优配置
    best_result = max(results, key=lambda x: x['test_pcc'])
    report.append(f"Best Configuration:")
    report.append(f"  Experiment: {best_result['experiment']}")
    report.append(f"  Speckle: {best_result['config']['speckle_size']}×{best_result['config']['speckle_size']}")
    report.append(f"  Model Output: {best_result['config']['model_output']}×{best_result['config']['model_output']}")
    report.append(f"  Test PCC: {best_result['test_pcc']:.4f}")
    report.append(f"  Test SSIM: {best_result['test_ssim']:.4f}")
    report.append("")
    
    # 关键发现
    report.append("Key Findings:")
    report.append("")
    
    # 1. 模型输出分辨率的影响
    s256_results = [r for r in results if r['config']['speckle_size'] == 256]
    s256_results.sort(key=lambda x: x['config']['model_output'])
    
    pcc_64 = s256_results[0]['test_pcc']
    pcc_128 = s256_results[1]['test_pcc']
    pcc_256 = s256_results[2]['test_pcc']
    
    gain_64_128 = (pcc_128 - pcc_64) / pcc_64 * 100
    gain_128_256 = (pcc_256 - pcc_128) / pcc_128 * 100
    
    report.append("1. Model Output Resolution Effect (Fixed Speckle 256×256):")
    report.append(f"   64→128: PCC improves by {gain_64_128:.1f}%")
    report.append(f"   128→256: PCC improves by {gain_128_256:.1f}%")
    
    if gain_128_256 < 2.0:
        report.append("   → Conclusion: 256 shows diminishing returns, 128 may be optimal")
    else:
        report.append("   → Conclusion: Higher output resolution consistently beneficial")
    report.append("")
    
    # 2. 散斑分辨率的影响
    m256_results = [r for r in results if r['config']['model_output'] == 256]
    m256_results.sort(key=lambda x: x['config']['speckle_size'])
    
    speckle_pcc_64 = m256_results[0]['test_pcc']
    speckle_pcc_128 = m256_results[1]['test_pcc']
    speckle_pcc_256 = m256_results[2]['test_pcc']
    
    loss_64 = (speckle_pcc_256 - speckle_pcc_64) / speckle_pcc_256 * 100
    loss_128 = (speckle_pcc_256 - speckle_pcc_128) / speckle_pcc_256 * 100
    
    report.append("2. Speckle Resolution Effect (Fixed Model 256×256):")
    report.append(f"   64×64: PCC loss = {loss_64:.1f}%")
    report.append(f"   128×128: PCC loss = {loss_128:.1f}%")
    report.append(f"   256×256: Baseline (best)")
    
    if loss_128 > 5.0:
        report.append("   → Conclusion: 256×256 speckle is critical, cannot downgrade")
    else:
        report.append("   → Conclusion: 128×128 speckle is acceptable for efficiency")
    report.append("")
    
    # 3. 实用建议
    report.append("3. Practical Recommendations:")
    
    # 找到性价比最优配置
    efficiency_score = []
    for r in results:
        # 效率分数 = PCC / (time * params^0.5)
        score = r['test_pcc'] / (r['time_hours'] * np.sqrt(r['parameters_M']))
        efficiency_score.append((r, score))
    
    efficiency_score.sort(key=lambda x: x[1], reverse=True)
    best_efficient = efficiency_score[0][0]
    
    report.append(f"   Best Quality: {best_result['experiment']} "
                 f"(PCC={best_result['test_pcc']:.4f})")
    report.append(f"   Best Efficiency: {best_efficient['experiment']} "
                 f"(PCC={best_efficient['test_pcc']:.4f}, "
                 f"Time={best_efficient['time_hours']:.1f}h)")
    report.append("")
    
    # 保存报告
    report_text = '\n'.join(report)
    print(report_text)
    
    with open(os.path.join(output_dir, 'Summary_Report.txt'), 'w') as f:
        f.write(report_text)
    
    print("\n✓ Summary report saved")


def main():
    """主函数"""
    # 设置路径
    base_dir = "./results"  # 下载的结果目录
    output_dir = "./figures"  # 图片输出目录
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading results...")
    results = load_all_results(base_dir)
    
    if len(results) == 0:
        print("Error: No results found!")
        print(f"Please make sure results are in: {base_dir}")
        return
    
    print(f"Found {len(results)} experiments\n")
    
    print("Generating visualizations...")
    
    # 生成所有图表
    plot_figure1_model_output_effect(results, output_dir)
    plot_figure2_speckle_resolution_effect(results, output_dir)
    plot_figure3_efficiency_quality_tradeoff(results, output_dir)
    plot_figure4_heatmap(results, output_dir)
    
    # 生成总结报告
    generate_summary_report(results, output_dir)
    
    print("\n" + "="*80)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
