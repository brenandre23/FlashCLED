import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Reconstructing the data from your image
data = {
    'Horizon': ['12-Week', '12-Week', '2-Week', '2-Week', '4-Week', '4-Week'],
    'Learner': ['lightgbm', 'xgboost', 'lightgbm', 'xgboost', 'lightgbm', 'xgboost'],
    'ROC_AUC': [0.965354, 0.969136, 0.960173, 0.965942, 0.970573, 0.965317],
    'PR_AUC': [0.066399, 0.095975, 0.096268, 0.130691, 0.105658, 0.115744],
    'Recall_10p': [0.921429, 0.917143, 0.880057, 0.896716, 0.927178, 0.89386],
    'RMSE': [9.958661, 10.44945, 11.77191, 11.8576, 10.0838, 10.80436]
}

df = pd.DataFrame(data)

# 2. Sorting Horizons logically (2 -> 4 -> 12) instead of alphabetically
horizon_order = ['2-Week', '4-Week', '12-Week']
df['Horizon'] = pd.Categorical(df['Horizon'], categories=horizon_order, ordered=True)
df = df.sort_values('Horizon')

# 3. Setup the plotting area
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison Across Horizons', fontsize=16, weight='bold')

# Metrics to plot
metrics = ['ROC_AUC', 'PR_AUC', 'Recall_10p', 'RMSE']
titles = ['ROC AUC (Higher is Better)', 'PR AUC (Higher is Better)', 
          'Recall @ 10% (Higher is Better)', 'RMSE (Lower is Better)']
colors = ['#1f77b4', '#2ca02c'] # Blue for LightGBM, Green for XGBoost

# 4. Loop to create subplots
bar_width = 0.35
indices = np.arange(len(horizon_order))

for i, ax in enumerate(axes.flat):
    metric = metrics[i]
    
    # Filter data for each learner
    lgbm_vals = df[df['Learner'] == 'lightgbm'][metric].values
    xgb_vals = df[df['Learner'] == 'xgboost'][metric].values
    
    # Create bars
    rects1 = ax.bar(indices - bar_width/2, lgbm_vals, bar_width, label='LightGBM', color='#a6cee3', edgecolor='black')
    rects2 = ax.bar(indices + bar_width/2, xgb_vals, bar_width, label='XGBoost', color='#1f78b4', edgecolor='black')
    
    # Formatting
    ax.set_title(titles[i], fontsize=12, weight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels(horizon_order)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars for clarity
    for rect in rects1 + rects2:
        height = rect.get_height()
        # Format decimal places differently for RMSE vs AUC
        fmt = '{:.1f}' if metric == 'RMSE' else '{:.3f}'
        ax.annotate(fmt.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# Add single legend
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
plt.show()