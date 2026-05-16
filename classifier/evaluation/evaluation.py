
import torch
import torchmetrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def evaluate_per_class(model, dataloader, device, num_classes, loss_fn):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if not type(outputs) == torch.Tensor:
                outputs = outputs.logits
            preds = torch.argmax(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            num_batches += 1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches

    # Calculate confusion matrix and class-wise metrics
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=range(num_classes))

    # Calculate specificity for each class
    specificity = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    metrics = {
        'loss': avg_loss,
        'precision': precision, 
        'recall': recall, 
        'f1': f1,
        'specificity': specificity
    }
    return metrics