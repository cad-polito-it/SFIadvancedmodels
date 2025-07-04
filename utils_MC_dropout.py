
import enum
from os import mkdir
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import os


def set_dropout_probability(model, dropout_probability):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            old_p = module.p
            module.p = dropout_probability
            print(f"Modificato dropout {name}: {old_p} → {dropout_probability}")



def dropout_layers_activation(model):
        # 3) Ri‑attiva SOLO i layer Dropout
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()             # Dropout ON
        elif isinstance(m, nn.BatchNorm2d):
            m.eval()              # (ridondante, già in eval, ma esplicito)
        



def check_dropout_and_batchnorm_status(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            print(f"[Dropout]     {name}: training = {module.training}")
        elif isinstance(module, nn.BatchNorm2d):
            print(f"[BatchNorm2d] {name}: training = {module.training}")
            
            


def mc_dropout_forward(model, x, forward_passes: int = 20):
    """
    Esegue più forward pass con dropout attivo per stimare
    distribuzione predittiva (mean / var) di un batch.

    Args:
        model           : rete neurale con layer Dropout interni
        x (Tensor)      : batch (B, C, H, W) o (B, F)
        forward_passes  : numero di campioni Monte Carlo

    Returns:
        mean_logits (B, C)
        var_logits  (B, C)
        mean_probs  (B, C)  # opzionale, se serve
        predictive_entropy (B,)  # opzionale, misura scalar di incertezza
    """
    logits_all = []
    
    MC_output = model(x)  # Esegui un forward pass per attivare i layer Dropout
    with torch.no_grad():            # disabilita il gradiente
        for _ in range(forward_passes):
            logits_all.append(model(x).unsqueeze(0))   # (1, B, C)
            # print(logits_all)

    logits_all = torch.cat(logits_all, dim=0)          # (T, B, C)
    mean_logits = logits_all.mean(dim=0)               # (B, C)
    var_logits = logits_all.var(dim=0, unbiased=False)

    # --- opzionale: distribuzione predittiva vera e propria -----------
    probs_all = torch.softmax(logits_all, dim=-1)      # (T, B, C)
    mean_probs = probs_all.mean(dim=0)                 # (B, C)
    # predictive entropy (scalar uncertainty per campione)
    predictive_entropy = -(mean_probs * mean_probs.log()).sum(dim=-1)  # (B,)

    return mean_logits, var_logits, mean_probs, predictive_entropy, MC_output



def dropout_inference(model, loader, device):
    clean_output_scores = list()
    clean_output_indices = list()
    clean_labels = list()
    counter = 0

    with torch.no_grad():
        pbar = tqdm(loader,
                colour='green',
                desc=f'Clean Run',
        )
        dataset_size = 0
        for batch_id, batch in enumerate(pbar):
            data, label = batch
            dataset_size = dataset_size + len(label)
            data = data.to(device)
            
            
            
            network_output = model(data)
            prediction = torch.topk(network_output, k=1)
            scores = network_output.cpu()
            indices = [int(fault) for fault in prediction.indices]
            
            clean_output_scores.append(scores)
            clean_output_indices.append(indices)
            clean_labels.append(label)
            
            counter = counter + 1


        elementwise_comparison = [label != index for labels, indices in zip(clean_labels, clean_output_indices) for label, index in zip(labels, indices)]          
        # Count the number of different elements
        num_different_elements = elementwise_comparison.count(True)
        
        print(f'device: {device}')

        print(f"The DNN wrong predicions are: {num_different_elements}")
        accuracy= (1 - num_different_elements/dataset_size)*100
        print(f"The final accuracy is: {accuracy}%")
        
        
def clean_dropout_MC_inference(model, loader, device, batch_size, max_images=1000):

    num_errors = 0
    pred_class = 0
    true_label = 0
    confidence = 0
    entropy = 0
    mean_logits_pred_class = 0
    var_pred_class = 0
    var_pred_class_n = 0
    total_var = 0
    all_variances = 0
    dataset_size = 0
    records = []
    batch_size = 1  # Set the batch size to 1 for processing one image at a time
    processed_images = 0

    pbar = tqdm(loader, colour='green', desc='MC Dropout Eval')

    for batch_id, batch in enumerate(pbar):
        
        data, label = batch
        data = data.to(device)
        label = label.to(device)  
        processed_images += batch_size
        
        # -------- MC Dropout multipass ------------
        mean_logits, var_logits, mean_probs, entropy, MC_out = mc_dropout_forward(
            model, data, forward_passes=20
        )

        # print(MC_out)
        
        preds       = mean_probs.argmax(dim=1)                # predizione finale
        confidences = mean_probs.max(dim=1).values            # confidenza finale
        dataset_size += label.size(0)
        num_errors   += (preds != label).sum().item()
        pbar.set_description(f'MC Dropout Eval - Errors: {num_errors} / {dataset_size}')
        
        # ------ Costruzione record sample per sample --------
        for i in range(data.size(0)):
        
            pred_class = preds[i].item()
            true_label = label[i].item()
            confidence = confidences[i].item()
            entropy_val = entropy[i].item()
            var_pred_class = var_logits[i, pred_class].item()
            mean_logits_pred_class = mean_logits[i, pred_class].item()  # Classe predetta
            var_pred_class_n = var_logits[i, pred_class].item() / var_logits[i].sum().item()
            total_var = var_logits[i].sum().item()
            # print('var_logit:',var_logits[i].sum().item())
            # exit(-1)
            all_variances = var_logits[i].tolist()  # Save all 10 variances as a list
            # print(var_pred_class_n)
            records.append({
                "pred_class": pred_class,
                "true_label": true_label,
                "confidence": confidence,
                "entropy": entropy_val,
                "mean logits": mean_logits_pred_class,  # Convert to list for DataFrame
                "var_pred_class": var_pred_class,
                "var_pred_class_n": var_pred_class_n,  # Normalized variance
                # "total_var": total_var,  # Total variance
                "all_variances": all_variances,  # Add the list of all variances
            })

    # Costruzione finale del DataFrame
    df = pd.DataFrame(records)
    df.to_csv("clean_mc_dropout_predictions.csv", index=False)

    print(f'Wrong predictions: {num_errors} / {dataset_size}')
    print(f'Accuracy        : {100 - (100 * num_errors / dataset_size):.2f}%')




def faulty_dropout_MC_inference(model, loader, device, fault_id):
    
    # Create the directory if it does not exist
    MC_output_dir = "MC_Dropout_Faulty_Results"
    if not os.path.exists(MC_output_dir):
        os.makedirs(MC_output_dir)
        
    num_errors = 0
    pred_class = 0
    true_label = 0
    confidence = 0
    entropy = 0
    mean_logits_pred_class = 0
    var_pred_class = 0
    var_pred_class_n = 0
    total_var = 0
    all_variances = 0
    dataset_size = 0
    records = []
    batch_size = 1  # Set the batch size to 1 for processing one image at a time

    pbar = tqdm(loader, colour='green', desc='MC Dropout Eval')

    for data, label in pbar:
        
        data = data.to(device)
        label = label.to(device)  
        
        # -------- MC Dropout multipass ------------
        mean_logits, var_logits, mean_probs, entropy, MC_output = mc_dropout_forward(
            model, data, forward_passes=20
        )

        preds       = mean_probs.argmax(dim=1)                # predizione finale
        confidences = mean_probs.max(dim=1).values            # confidenza finale
        dataset_size += label.size(0)
        num_errors   += (preds != label).sum().item()
        pbar.set_description(f'MC Dropout Eval - Errors: {num_errors} / {dataset_size}')
        
        # ------ Costruzione record sample per sample --------
        for i in range(data.size(0)):
        
            pred_class = preds[i].item()
            true_label = label[i].item()
            confidence = confidences[i].item()
            entropy_val = entropy[i].item()
            var_pred_class = var_logits[i, pred_class].item()
            mean_logits_pred_class = mean_logits[i, pred_class].item()  # Classe predetta
            var_pred_class_n = var_logits[i, pred_class].item() / var_logits[i].sum().item()
            total_var = var_logits[i].sum().item()
            all_variances = var_logits[i].tolist()  # Save all 10 variances as a list
            # print(var_pred_class_n)
            records.append({
                "fault_id": fault_id,  # Add fault_id to the record
                "pred_class": pred_class,
                "true_label": true_label,
                "confidence": confidence,
                "entropy": entropy_val,
                "mean logits": mean_logits_pred_class,  # Convert to list for DataFrame
                "var_pred_class": var_pred_class,
                "var_pred_class_n": var_pred_class_n,  # Normalized variance
                # "total_var": total_var,  # Total variance
                "all_variances": all_variances,  # Add the list of all variances
            })

    # Costruzione finale del DataFrame
    df = pd.DataFrame(records)
    df.to_csv(f"{MC_output_dir}/fault_{fault_id}_clean_mc_dropout_predictions.csv", index=False)

    # print(f'Wrong predictions: {num_errors} / {dataset_size}')
    # print(f'Accuracy        : {100 - (100 * num_errors / dataset_size):.2f}%')
    
    return MC_output