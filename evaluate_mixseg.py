import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置更高质量的图表参数
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 读取CSV文件
def analyze_metrics(csv_path):
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 根据文件名前缀提取数据集名称
    df['Dataset'] = df['Filename'].apply(lambda x: x.split('_')[0])
    
    # 计算每个数据集的Dice和IoU均值和标准差
    dataset_stats = df.groupby('Dataset').agg({
        'Dice': ['mean', 'std', 'count'],
        'IoU': ['mean', 'std']
    })
    
    # 重命名列
    dataset_stats.columns = ['Dice_Mean', 'Dice_Std', 'Count', 'IoU_Mean', 'IoU_Std']
    dataset_stats = dataset_stats.reset_index()
    
    # 保存dataset_stats到CSV
    stats_dir = os.path.dirname(csv_path)
    dataset_stats.to_csv(os.path.join(stats_dir, 'dataset_performance.csv'), index=False)
    
    # 创建Dice可视化
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='Dataset', y='Dice', data=df, hue='Dataset', legend=False)
    
    # 在箱形图上添加均值和标准差标注
    for i, ds in enumerate(dataset_stats['Dataset']):
        mean = dataset_stats.loc[dataset_stats['Dataset'] == ds, 'Dice_Mean'].values[0]
        std = dataset_stats.loc[dataset_stats['Dataset'] == ds, 'Dice_Std'].values[0]
        count = dataset_stats.loc[dataset_stats['Dataset'] == ds, 'Count'].values[0]
        ax.text(i, 0.1, f'n={count}\nμ={mean:.3f}\nσ={std:.3f}', 
                horizontalalignment='center', size='small', color='black', weight='semibold')
    
    plt.title('Dice Score Distribution by Dataset')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'dice_distribution.pdf'), bbox_inches='tight')
    
    # 创建IoU可视化
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='Dataset', y='IoU', data=df, hue='Dataset', legend=False)
    
    # 在箱形图上添加均值和标准差标注
    for i, ds in enumerate(dataset_stats['Dataset']):
        mean = dataset_stats.loc[dataset_stats['Dataset'] == ds, 'IoU_Mean'].values[0]
        std = dataset_stats.loc[dataset_stats['Dataset'] == ds, 'IoU_Std'].values[0]
        count = dataset_stats.loc[dataset_stats['Dataset'] == ds, 'Count'].values[0]
        ax.text(i, 0.1, f'n={count}\nμ={mean:.3f}\nσ={std:.3f}', 
                horizontalalignment='center', size='small', color='black', weight='semibold')
    
    plt.title('IoU Score Distribution by Dataset')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'iou_distribution.pdf'), bbox_inches='tight')
    
    # 创建一个综合可视化：均值和标准差的条形图
    plt.figure(figsize=(14, 10))
    
    # 子图1: Dice
    plt.subplot(2, 1, 1)
    bar_positions = np.arange(len(dataset_stats))
    bar_width = 0.8
    
    bars = plt.bar(bar_positions, dataset_stats['Dice_Mean'], bar_width, 
                   yerr=dataset_stats['Dice_Std'], capsize=5, 
                   color=sns.color_palette('Set3', len(dataset_stats)))
    
    # 在条形上添加标注
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = dataset_stats.iloc[i]['Count']
        plt.text(bar.get_x() + bar.get_width()/2., height + dataset_stats.iloc[i]['Dice_Std'] + 0.02,
                f'{height:.3f}±{dataset_stats.iloc[i]["Dice_Std"]:.3f}\nn={count}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xticks(bar_positions, dataset_stats['Dataset'])
    plt.ylabel('Dice Score')
    plt.title('Mean Dice Score by Dataset with Standard Deviation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 子图2: IoU
    plt.subplot(2, 1, 2)
    
    bars = plt.bar(bar_positions, dataset_stats['IoU_Mean'], bar_width, 
                   yerr=dataset_stats['IoU_Std'], capsize=5, 
                   color=sns.color_palette('Set3', len(dataset_stats)))
    
    # 在条形上添加标注
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + dataset_stats.iloc[i]['IoU_Std'] + 0.02,
                f'{height:.3f}±{dataset_stats.iloc[i]["IoU_Std"]:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xticks(bar_positions, dataset_stats['Dataset'])
    plt.ylabel('IoU Score')
    plt.title('Mean IoU Score by Dataset with Standard Deviation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'mean_scores_comparison.pdf'), bbox_inches='tight')
    
    print(f"分析完成，图表已保存到 {stats_dir}")
    
    return dataset_stats

if __name__ == "__main__":
    # 找到CSV文件
    csv_file = "logs/train_mixseg_resize_512_BUSC,UDIAT,STU,OASBUD,BrEaST_2025_04_26_10_37_33/Log/individual_metrics.csv"
    
    # 如果文件不存在，尝试查找其他可能的路径
    if not os.path.exists(csv_file):
        for root, dirs, files in os.walk("logs"):
            for file in files:
                if file == "individual_metrics.csv":
                    csv_file = os.path.join(root, file)
                    break
    
    # 确保文件存在
    if not os.path.exists(csv_file):
        print(f"找不到CSV文件: {csv_file}")
    else:
        # 执行分析
        stats = analyze_metrics(csv_file)
        print("各数据集性能统计:")
        print(stats)
