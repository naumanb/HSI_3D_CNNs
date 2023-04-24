import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Save the classification report and confusion matrix visualization
def save_outputs(y_true, y_pred, output_folder, file_name, architecture_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save classification report
    report = classification_report(y_true, y_pred)
    report_file_name = f"{architecture_name}_{file_name}_report.txt"
    with open(os.path.join(output_folder, report_file_name), 'w') as f:
        f.write(report)

    # Save confusion matrix visualization
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    cm_file_name = f"{architecture_name}_{file_name}_confusion_matrix.png"
    plt.savefig(os.path.join(output_folder, cm_file_name))
    plt.clf()