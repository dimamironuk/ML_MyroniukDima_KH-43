import plotly.graph_objects as go

def plot_interactive_roc(roc_data):
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, (fpr, tpr, auc_score)) in enumerate(roc_data.items()):
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name=f'{name} (AUC = {auc_score:.4f})',
            line=dict(color=colors[i % len(colors)], width=2.5)
        ))
        
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', 
        name='Random Baseline',
        line=dict(color='black', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Interactive ROC Curves Comparison',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        template='plotly_white',
        width=900, height=600,
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95)
    )
    fig.show()

def plot_interactive_pr(pr_data):
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, (recall, precision, pr_auc_score)) in enumerate(pr_data.items()):
        fig.add_trace(go.Scatter(
            x=recall, y=precision, mode='lines', 
            name=f'{name} (PR-AUC = {pr_auc_score:.4f})',
            line=dict(color=colors[i % len(colors)], width=2.5)
        ))
        
    fig.update_layout(
        title='Interactive Precision-Recall Curves Comparison',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_white',
        width=900, height=600,
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95)
    )
    fig.show()