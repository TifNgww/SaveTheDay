import argparse
import itertools
import sys
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
#from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from pyfasttext import FastText




def load_notes_labels(txt_file_path):
    if txt_file_path == None:
        return
    file = open(txt_file_path,"r")
    label=[]
    notes=[]
    for line in file:
        fields = line.split(" ",1)
        label.append(fields[0][9:])
        notes.append(fields[1])
    file.close()
    return notes,label
                                               
def load_fasttext_model(model_bin_path):
    if model_bin_path == None:
        return
    return FastText(model_bin_path)
                                               
                                               
def get_pred_labels(model,notes):
    pred_labels = model.predict(notes)
    pred_labels  = [item for sublist in pred_labels for item in sublist]
    return pred_labels
                                              
                                               
 
def print_eval_to_txt(log_path,true_labels, pred_labels,txt_file_path):
    correct=sum(1 for a, b in zip(true_labels, pred_labels) if a == b)
    true_labels_unique = np.unique(true_labels)
    pred_labels_unique = np.unique(pred_labels)
    target_names=np.unique(true_labels + pred_labels)
    sys.stdout = open(log_path, "w")
    print("# of Catergories: " + str(len(target_names)))
    print("N: " + str(len(true_labels)))
    print("Accuracy: " + str(round(correct / len(true_labels),2)))
    if (set(true_labels_unique) - set(pred_labels_unique)):
        print("Categories " + str( list(set(true_labels_unique) - set(pred_labels_unique))) + " have no predicted samples in "+txt_file_path)
    if (set(pred_labels_unique) - set(true_labels_unique)):
        print("Categories " + str(list(set(pred_labels_unique) - set(true_labels_unique))) + " have no true samples in "+ txt_file_path)
    print(classification_report(true_labels, pred_labels, target_names))
    return target_names



def plot_confusion_matrix(true_labels, pred_labels, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm=confusion_matrix(true_labels, pred_labels)
    
    if normalize:
        bot= cm.sum(axis=1)[:, np.newaxis]
        bot[bot == 0] = 1
        cm = 100*cm.astype('float') / bot
        #print("Normalized confusion matrix")
    #else:   
        #print('Confusion matrix, without normalization')
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def confusion_plot_to_pdf(pdf_path,true_labels, pred_labels,target_names):
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    fig=plt.figure(figsize=(16,20))
    plot_confusion_matrix(true_labels, pred_labels,
                        classes=target_names,
                    title='Confusion matrix, with normalization (Percentage)')
    
    pdf.savefig()
    plt.gcf().clear()
    pdf.close()


                                              
                                               
                                               

def main(params):
    model=load_fasttext_model(params.model_bin_path)
    notes,true_labels= load_notes_labels(params.txt_file_path)
    pred_labels= get_pred_labels(model=model,notes=notes)
    target_names= print_eval_to_txt(params.log_path,true_labels=true_labels, pred_labels=pred_labels,txt_file_path=params.txt_file_path)
    confusion_plot_to_pdf(params.pdf_path,true_labels=true_labels, pred_labels=pred_labels,target_names=target_names)
                                               



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_bin_path', type=str,
                        default='/home/ec2-user/fastText-0.1.0/model_es.bin',
                        help='Path to model.bin')
    parser.add_argument('--txt_file_path', type=str,
                        default='/home/ec2-user/fastText-0.1.0/cm_zendesk_es.test.txt',
                        help='Path to txt')
    parser.add_argument('--log_path', type=str,
                        default='log.txt',
                        help='Path to log')
    parser.add_argument('--pdf_path', type=str,
                        default='confusion_matrix.pdf',
                        help='Path to pdf')
    params, _ = parser.parse_known_args()
    main(params)


