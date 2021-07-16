from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np

def bal_acc(y_true, y_pred):
    
    
    return balanced_accuracy_score(y_true, y_pred)

def class_report(y_true, y_pred):
    
    print(classification_report(y_true, y_pred))
    
def plot_confusion_matrix(y_true, y_pred, classes = [0, 1, 2, 3, 4, 5, 6, 7, 8],
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes = ["drink","eat","groom","hang","sniff","rear","rest","walk","eathand"]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def slackify(y_true, y_pred, slack=20):
    y_pred_slack = []
    for i,yp in enumerate(y_pred):
            
        
        #print(int(max(0,i-slack/2)),int(min(i+1+slack/2, len(y_true))))
        if yp in y_true[int(max(0,i-slack/2)) : int(min(i+1+slack/2, len(y_true)))]:
            y_pred_slack.append(y_true[i])
        else:
            y_pred_slack.append(yp)

    return y_pred_slack
