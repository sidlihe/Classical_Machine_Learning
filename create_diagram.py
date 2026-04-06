import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(8, 11.5, 'Classical_Machine_Learning - Project Structure', 
        fontsize=20, fontweight='bold', ha='center')

# Color definitions
colors = {
    'core': '#e3f2fd',
    'data': '#f3e5f5',
    'model': '#ffe0b2',
    'train': '#e0f2f1',
    'eval': '#fce4ec',
    'devops': '#e8f5e9'
}

def draw_box(ax, x, y, w, h, title, items, color, edge_color='black'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                         edgecolor=edge_color, facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.3, title, fontsize=11, fontweight='bold', 
            ha='center', va='top')
    y_offset = y + h - 0.7
    for item in items:
        ax.text(x + w/2, y_offset, item, fontsize=9, ha='center', va='top')
        y_offset -= 0.25

# Core Foundation Layer
ax.text(0.5, 10.8, 'Core Foundation', fontsize=12, fontweight='bold')
draw_box(ax, 0.5, 9.8, 2, 0.9, 'Logger', ['Centralized Logging', 'File + Console'], colors['core'], '#1976d2')
draw_box(ax, 2.7, 9.8, 2, 0.9, 'Config', ['YAML Management', 'Env Overrides'], colors['core'], '#1976d2')
draw_box(ax, 4.9, 9.8, 2, 0.9, 'Reproducibility', ['Seed Management', 'Deterministic'], colors['core'], '#1976d2')
draw_box(ax, 7.1, 9.8, 2, 0.9, 'Helpers', ['Utilities', 'Path Management'], colors['core'], '#1976d2')

# Data Pipeline Layer
ax.text(0.5, 9.3, 'Data Pipeline', fontsize=12, fontweight='bold')
draw_box(ax, 0.5, 8, 2.5, 1.2, 'DataLoader', ['CSV, Parquet, JSON', 'Loading & Caching'], colors['data'], '#7b1fa2')
draw_box(ax, 3.2, 8, 2.5, 1.2, 'DataValidator', ['8+ Data Checks', 'Quality Assurance'], colors['data'], '#7b1fa2')
draw_box(ax, 5.9, 8, 2.5, 1.2, 'Features', ['Polynomial, Interaction', 'Scaling & Selection'], colors['data'], '#7b1fa2')

# ML Models Layer
ax.text(0.5, 7.5, 'ML Models (20+)', fontsize=12, fontweight='bold')
draw_box(ax, 0.5, 5.8, 1.8, 1.6, 'Classical ML', ['Logistic Regression', 'SVM, KNN', 'Trees, Naive Bayes', '(5+ models)'], colors['model'], '#f57c00')
draw_box(ax, 2.5, 5.8, 1.8, 1.6, 'Ensemble', ['Random Forest', 'AdaBoost, Voting', 'Stacking', '(6+ models)'], colors['model'], '#f57c00')
draw_box(ax, 4.5, 5.8, 1.8, 1.6, 'Gradient Boost', ['XGBoost', 'LightGBM', 'CatBoost', '(3 models)'], colors['model'], '#f57c00')
draw_box(ax, 6.5, 5.8, 1.8, 1.6, 'Model Registry', ['Versioning', 'MLflow Integration', 'Artifact Mgmt', '(Production)'], colors['model'], '#f57c00')

# Training & Evaluation Layer
ax.text(0.5, 5.3, 'Training & Evaluation', fontsize=12, fontweight='bold')
draw_box(ax, 0.5, 3.5, 1.8, 1.7, 'Trainer', ['Orchestration', 'Model Compare', 'HPO Ready', 'GridSearch'], colors['train'], '#00897b')
draw_box(ax, 2.5, 3.5, 1.8, 1.7, 'Validation', ['K-Fold', 'TimeSeries', 'Hold-out', '(3 strategies)'], colors['train'], '#00897b')
draw_box(ax, 4.5, 3.5, 1.8, 1.7, 'Metrics', ['Accuracy, F1', 'ROC-AUC', 'Confusion Matrix', '(10+ metrics)'], colors['eval'], '#c2185b')
draw_box(ax, 6.5, 3.5, 1.8, 1.7, 'Visualization', ['ROC Curves', 'Feature Importance', 'Confusion Matrix', '(Multiple)'], colors['eval'], '#c2185b')

# Production & DevOps Layer
ax.text(0.5, 3, 'Production & DevOps', fontsize=12, fontweight='bold')
draw_box(ax, 0.5, 1.2, 2.2, 1.7, 'Scripts', ['train.py (Complete)', 'evaluate.py', 'predict.py', 'Full Pipeline'], colors['devops'], '#388e3c')
draw_box(ax, 3, 1.2, 2.2, 1.7, 'CI/CD', ['GitHub Actions', 'Testing', 'Training', 'Linting'], colors['devops'], '#388e3c')
draw_box(ax, 5.5, 1.2, 2.2, 1.7, 'DVC Pipeline', ['Data Management', 'Prepare Stage', 'Train Stage', 'Evaluate Stage'], colors['devops'], '#388e3c')

# Right side - Additional Info
ax.text(8.5, 10.8, 'Project Highlights', fontsize=12, fontweight='bold')
draw_box(ax, 8.5, 9.5, 3.5, 1.2, 'Statistics', ['3,500+ Lines Code', '14 Core Modules', '20+ ML Models', 'Production Grade'], '#f5f5f5', '#333')

ax.text(8.5, 8.9, 'Configuration', fontsize=12, fontweight='bold')
draw_box(ax, 8.5, 7.3, 3.5, 1.5, 'YAML Config', ['Data Settings', 'Preprocessing Options', 'Model Hyperparameters', 'Training Strategy'], '#f5f5f5', '#333')

ax.text(8.5, 6.8, 'Storage', fontsize=12, fontweight='bold')
draw_box(ax, 8.5, 5.2, 3.5, 1.5, 'Directories', ['data/ (raw, processed)', 'models/ (registry)', 'logs/ (tracking)', 'mlruns/ (MLflow)'], '#f5f5f5', '#333')

ax.text(8.5, 4.7, 'Features', fontsize=12, fontweight='bold')
features = [
    'Modular Architecture',
    'Centralized Logging',
    'YAML Configuration',
    'Data Validation',
    'Feature Engineering',
    'Model Versioning',
    'Experiment Tracking',
    'GitHub Actions CI/CD'
]
y_pos = 4.3
for feat in features:
    ax.text(8.7, y_pos, f'+ {feat}', fontsize=9)
    y_pos -= 0.35

# Add some arrows to show data flow
arrow_props = dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.6)
ax.annotate('', xy=(2, 8), xytext=(1.5, 9.8), arrowprops=arrow_props)
ax.annotate('', xy=(4, 8), xytext=(3.5, 9.8), arrowprops=arrow_props)

plt.tight_layout()
plt.savefig('PROJECT_STRUCTURE_DIAGRAM.png', dpi=150, bbox_inches='tight', facecolor='white')
print('+ PNG diagram created successfully: PROJECT_STRUCTURE_DIAGRAM.png')

# Convert to JPG
from PIL import Image
img = Image.open('PROJECT_STRUCTURE_DIAGRAM.png')
img.convert('RGB').save('PROJECT_STRUCTURE_DIAGRAM.jpg', 'JPEG', quality=95)
print('+ JPG diagram created successfully: PROJECT_STRUCTURE_DIAGRAM.jpg')

import os
png_size = os.path.getsize('PROJECT_STRUCTURE_DIAGRAM.png') / 1024
jpg_size = os.path.getsize('PROJECT_STRUCTURE_DIAGRAM.jpg') / 1024
print(f'PNG: {png_size:.1f} KB | JPG: {jpg_size:.1f} KB')
