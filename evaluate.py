import torch
import matplotlib.pyplot as plt
import monai
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from config import DEVICE, WIN_SIZE

def evaluate_and_interpret(model, test_loader, test_ds, dict_label2num, dict_num2label):
    model.eval()
    y_pred_list = torch.tensor([], dtype=torch.float32, device=DEVICE)
    y_list = torch.tensor([], dtype=torch.long, device=DEVICE)

    # 1. Classification Metrics
    for val_data in test_loader:
        val_images = val_data[0].to(DEVICE)
        val_labels = val_data[1].to(DEVICE).argmax(dim=1)

        with torch.no_grad():
            outputs = model(val_images)
            y_pred_list = torch.cat([y_pred_list, outputs.argmax(dim=1)], dim=0)
            y_list = torch.cat([y_list, val_labels], dim=0)

    print(classification_report(
        y_list.cpu().numpy(), y_pred_list.cpu().numpy(), target_names=list(dict_label2num.keys())
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_list.cpu().numpy(), y_pred_list.cpu().numpy(), normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(dict_label2num.keys()))
    fig, ax = plt.subplots(1, 1, facecolor="white")
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    plt.show()

    # 2. Interpretability (GradCAM & Occlusion Sensitivity)
    cam = monai.visualize.GradCAM(nn_module=model, target_layers="class_layers.relu")
    occ_sens = monai.visualize.OcclusionSensitivity(nn_module=model, mask_size=12, n_batch=1)
    
    the_slice = test_ds[0][0].shape[-1] // 2
    occ_sens_b_box = [-1, -1, -1, -1, the_slice - 1, the_slice]

    print("GradCAM and Occlusion setup complete. Proceeding with visualization...")
