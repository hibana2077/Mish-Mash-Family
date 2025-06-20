"""
Logging and evaluation utilities.
"""
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def save_metrics(history, save_path):
    """
    Save training metrics to CSV.
    
    Args:
        history: List of training metrics
        save_path: Path to save the CSV file
    """
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False)


def save_classification_report(y_true, y_pred, class_names, save_path):
    """
    Save sklearn classification report to file.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the report
    """
    # Generate classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names if class_names else None,
        output_dict=True,
        zero_division=0
    )
    
    # Save as JSON-like format
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(save_path)
    
    # Also save as text format
    text_report = classification_report(
        y_true, y_pred,
        target_names=class_names if class_names else None,
        zero_division=0
    )
    
    text_save_path = save_path.replace('.csv', '.txt')
    with open(text_save_path, 'w') as f:
        f.write(text_report)


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Save confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    if class_names and len(class_names) <= 20:  # Only show labels if not too many classes
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_output_filename(model_name, dataset_name, activation_type, extension):
    """
    Create output filename with the format {model_name}-{dataset}-{activation-type}.{extension}
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        activation_type: Type of activation function
        extension: File extension
        
    Returns:
        str: Formatted filename
    """
    return f"{model_name}-{dataset_name}-{activation_type}.{extension}"