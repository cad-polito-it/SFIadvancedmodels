import csv
import enum
from http.client import SEE_OTHER
import os
import shutil
import time
import math
from datetime import timedelta
import copy
from unittest import loader
# from turtle import mode

import SETTINGS
import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from tqdm import tqdm

from faultManager.NeuronFault import NeuronFault
from faultManager.WeightFaultInjector import WeightFaultInjector

from typing import List, Union
from utils_MC_dropout import faulty_dropout_MC_inference
from utils_MC_dropout import mc_dropout_forward
import pandas as pd


class FaultInjectionManager:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 loader: DataLoader,
                 clean_output: torch.Tensor,
                 injectable_modules: List[Union[Module, List[Module]]] = None):

        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.device = device

        self.clean_output = clean_output
        self.faulty_output = list()

        # The folder used for the logg
        self.__log_folder = f'log/{self.network_name}/batch_{self.loader.batch_size}'

        # The folder where to save the output
        self.__faulty_output_folder = SETTINGS.FAULTY_OUTPUT_FOLDER

        # The number of total inferences and the number of skipped inferences
        self.skipped_inferences = 0
        self.total_inferences = 0

        # The weight fault injector
        self.weight_fault_injector = WeightFaultInjector(self.network)

        # The list of injectable module, used only for neuron fault injection
        self.injectable_modules = injectable_modules


    def run_clean_campaign(self):

        pbar = tqdm(self.loader,
                    desc='Clean Inference',
                    colour='green')

        for batch_id, batch in enumerate(pbar):
            data, _ = batch
            data = data.to(self.device)

            self.network(data)

    
    def restore_fault(self):
        """Ripristina il valore originale se è stato modificato."""
        if self.golden_value is not None and self.layer_name is not None:
            self.network.state_dict()[self.layer_name][self.tensor_index] = self.golden_value
            # print(f"Restored {self.layer_name} at index {self.tensor_index} to {self.golden_value}")
            # Dopo il ripristino, svuotiamo la memoria per evitare problemi
            self.golden_value = None
            self.layer_name = None
            self.tensor_index = None
        else:
            print("No fault to restore!")

    def run_faulty_campaign_on_weight(self,
                                      fault_model: str,
                                      fault_list: list,
                                      first_batch_only: bool = False,
                                      force_n: int = None,
                                      save_output: bool = False,
                                      save_ofm: bool = False,
                                      ofm_folder: str = None) -> (str, int):
        """
        Run a faulty injection campaign for the network. If a layer name is specified, start the computation from that
        layer, loading the input feature maps of the previous layer
        :param fault_model: The faut model for the injection
        :param fault_list: list of fault to inject. One of ['byzantine_neuron', 'stuck-at_params']
        :param first_batch_only: Default False. Debug parameter, if set run the fault injection campaign on the first
        batch only
        :param force_n: Default None. If set, inject only force_n faults in the network
        :param save_output: Default False. Whether to save the output of the network or not
        :param save_ofm: Default False. Whether to save the ofm of the injectable layers
        :param ofm_folder: Default None. The folder where to save the ofms if save_fm is true
        :return: A tuple formed by : (i) a string containing the formatted time elapsed from the beginning to the end of
        the fault injection campaign, (ii) an integer measuring the average memory occupied (in MB)
        """

        self.skipped_inferences = 0
        self.total_inferences = 0

        total_different_predictions = 0
        total_predictions = 0

        average_memory_occupation = 0
        total_iterations = 1

        with torch.no_grad():

            if force_n is not None:
                fault_list = fault_list[:force_n]

            # Order the fault list to speed up the injection
            # This is also important to avoid differences between a
            fault_list = sorted(fault_list, key=lambda x: x.injection)

            # Start measuring the time elapsed
            start_time = time.time()

            # The dict measuring the accuracy of each batch
            accuracy_dict = dict()

            # Cycle all the batches in the data loader
            for batch_id, batch in enumerate(self.loader):
                data, target = batch
                data = data.to(self.device)

                # The list of the accuracy of the network for each fault
                accuracy_batch_dict = dict()
                accuracy_dict[batch_id] = accuracy_batch_dict

                faulty_prediction_dict = dict()
                batch_clean_prediction_scores = [float(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).values]
                batch_clean_prediction_indices = [int(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).indices]


                # Inject all the faults in a single batch
                pbar = tqdm(fault_list,
                            colour='green',
                            desc=f'FI on b {batch_id}',
                            ncols=shutil.get_terminal_size().columns * 2)
                for fault_id, fault in enumerate(pbar):
                    # fault._print()
                    # Change the description of the progress bar
                    # if fault_dropping and fault_delayed_start:
                    #     pbar.set_description(f'FI (w/ drop & delayed) on b {batch_id}')
                    # elif fault_dropping:
                    #     pbar.set_description(f'FI (w/ drop) on b {batch_id}')
                    # elif fault_delayed_start:
                    #     pbar.set_description(f'FI (w/ delayed) on b {batch_id}')

                    # ----------------------------- #

                    # Inject faults
                    if fault_model == 'byzantine_neuron':
                        injected_layer = self.__inject_fault_on_neuron(fault=fault)
                    elif fault_model == 'stuck-at_params':
                        self.__inject_fault_on_weight(fault, fault_mode='stuck-at')
                    else:
                        raise ValueError(f'Invalid fault model {fault_model}')

                    # Reset memory occupation stats
                    torch.cuda.reset_peak_memory_stats()

                    # If you have to save the ifm, update the file names
                    if save_ofm:
                        for injectable_module in self.injectable_modules:
                            injectable_module.ifm_path = f'{ofm_folder}/fault_{fault_id}_batch_{batch_id}_layer_{injectable_module.layer_name}'

                  
                    faulty_scores, faulty_indices, different_predictions = self.__run_inference_on_batch(batch_id=batch_id,
                                                                                                         data=data)

                    # Measure the memory occupation
                    memory_occupation = (torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved()) // (1024**2)
                    average_memory_occupation = ((total_iterations - 1) * average_memory_occupation + memory_occupation) // total_iterations

                    # If fault prediction is None, the fault had no impact. Use golden predictions
                    if faulty_indices is None:
                        faulty_scores = self.clean_output[batch_id]
                        faulty_indices = batch_clean_prediction_indices

                    # Measure the accuracy of the batch
                    accuracy_batch_dict[fault_id] = float(torch.sum(target.eq(torch.tensor(faulty_indices)))/len(target))

                    # Move the scores to the gpu
                    faulty_scores = faulty_scores.detach().cpu()

                    faulty_prediction_dict[fault_id] = tuple(zip(faulty_indices, faulty_scores))
                    total_different_predictions += different_predictions

                    # Store the faulty prediction if the option is set
                    if save_output:
                        self.faulty_output.append(faulty_scores.numpy())

                    # Measure the loss in accuracy
                    total_predictions += len(batch[0])
                    different_predictions_percentage = 100 * total_different_predictions / total_predictions
                    pbar.set_postfix({'Different': f'{different_predictions_percentage:.6f}%',
                                      'Skipped': f'{100*self.skipped_inferences/self.total_inferences:.2f}%',
                                      'Avg. memory': f'{average_memory_occupation} MB'}
                                     )

                    # Clean the fault
                    if fault_model == 'byzantine_neuron':
                        injected_layer.clean_fault()
                    elif fault_model == 'stuck-at_params':
                        self.weight_fault_injector.restore_golden()
                    else:
                        raise ValueError(f'Invalid fault model {fault_model}')

                    # Increment the iteration count
                    total_iterations += 1

                # Log the accuracy of the batch
                os.makedirs(f'{self.__log_folder}/{fault_model}', exist_ok=True)
                log_filename = f'{self.__log_folder}/{fault_model}/batch_{batch_id}.csv'
                with open(log_filename, 'w') as log_file:
                    log_writer = csv.writer(log_file)
                    log_writer.writerows(accuracy_batch_dict.items())

                # Save the output to file if the option is set
                if save_output:
                    os.makedirs(f'{self.__faulty_output_folder}/{fault_model}', exist_ok=True)
                    np.save(f'{self.__faulty_output_folder}/{fault_model}/batch_{batch_id}', self.faulty_output)
                    self.faulty_output = list()

                # End after only one batch if the option is specified
                if first_batch_only:
                    break


        # Measure the average accuracy
        average_accuracy_dict = dict()
        for fault_id in range(len(fault_list)):
            fault_accuracy = np.average([accuracy_batch_dict[fault_id] for _, accuracy_batch_dict in accuracy_dict.items()])
            average_accuracy_dict[fault_id] = float(fault_accuracy)

        # Final log
        os.makedirs(f'{self.__log_folder}/{fault_model}', exist_ok=True)
        log_filename = f'{self.__log_folder}/{fault_model}/all_batches.csv'
        with open(log_filename, 'w') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerows(average_accuracy_dict.items())


        elapsed = math.ceil(time.time() - start_time)

        return str(timedelta(seconds=elapsed)), average_memory_occupation


    

    
    def run_faulty_campaign_on_weight_segmentation(self,
                                      fault_model: str,
                                      fault_list: list,
                                      first_batch_only: bool = False,
                                      force_n: int = None,
                                      save_output: bool = False,
                                      save_ofm: bool = False,
                                      ofm_folder: str = None) -> (str, int):
        
        self.skipped_inferences = 0
        self.total_inferences = 0
        
        os.makedirs(SETTINGS.FI_ANALYSIS_PATH, exist_ok=True)
        
        csv_file = open(f'{SETTINGS.FI_ANALYSIS_PATH}/results.csv', 'a')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image_ID', 'Fault_ID', 'same_pixel_gold','same_pixel_label',*(f"IOU_class_{i}_gold" for i in range(21)),*(f"IOU_class_{i}_label" for i in range(21))])      

        golden_output = np.load(f'{SETTINGS.CLEAN_OUTPUT_FOLDER}/all_batches.npy')
        golden_output = torch.tensor(golden_output)
        
        with torch.no_grad():
            
            if force_n is not None:
                fault_list = fault_list[:force_n]
                
            fault_list = sorted(fault_list, key=lambda x: x.injection)
            
            
            pbar2 = tqdm(fault_list,
                        colour='green',
                        desc=f'Fault Injection',
                        ncols=shutil.get_terminal_size().columns)
            
            # Create a CSV file to save the data
           
            for fault_id, fault in enumerate(pbar2):
                
        
                if fault_model == 'stuck-at_params':
                    self.__inject_fault_on_weight(fault, fault_mode='stuck-at')
                else:
                    raise ValueError(f'Invalid fault model {fault_model}')
                
                torch.cuda.reset_peak_memory_stats()
                
                batch_id = 0
                correctPixels = 0
                numclass = 21
                pbar = tqdm(self.loader, 
                        colour='green',
                        desc=f'fault_id {fault_id}',
                        ncols=shutil.get_terminal_size().columns * 2)  # progress bar
                
                for img, label in pbar:
                    
                    img = img.to(self.device)
                    label = label.to(self.device)
                    label = label.squeeze(1)
                                        
                    
                    output = self.network(img)["out"]
                    faulty_pred = output.argmax(axis=1)
                    
                    
                    actual_batch_size = faulty_pred.size(0) 
                    golden_pred = golden_output[batch_id][:actual_batch_size].to(self.device)
                     
                    
                    diff = golden_pred == faulty_pred
                    correctPixels = diff.sum(axis=[1,2])

                    diff2 = label == faulty_pred
                    correctPixels2 = diff2.sum(axis=[1,2])
           
                    
                    
              
                    # IoU between faulty and golden
                    ious = torch.zeros((numclass, actual_batch_size))
                    for cls in range(numclass):
                        clsPred = faulty_pred == cls
                        clsLab = golden_pred == cls
                        inter = torch.logical_and(clsPred, clsLab).sum(axis=[1,2])
                        union = torch.logical_or(clsPred, clsLab).sum(axis=[1,2])
                        iou = inter/union
                        # print(iou.shape)
                        ious[cls] = iou
                        
                    ious2 = torch.zeros((numclass, actual_batch_size))
                    for cls2 in range(numclass):
                        clsPred2 = faulty_pred == cls2
                        clsLab2 = label == cls2
                  
                        inter2 = torch.logical_and(clsPred2, clsLab2).sum(axis=[1,2])
                        union2 = torch.logical_or(clsPred2, clsLab2).sum(axis=[1,2])
                        iou2 = inter2/union2
                        ious2[cls2] = iou2
                      
                    for i in range(actual_batch_size):
                        csv_writer.writerow([i+SETTINGS.BATCH_SIZE*batch_id, fault_id, correctPixels[i].tolist(), correctPixels2[i].tolist(), *(ious[:,i].tolist()), *(ious2[:,i].tolist())])                

                    batch_id += 1
                
                
                
                # Clean the fault    
                if fault_model == 'stuck-at_params':
                    self.weight_fault_injector.restore_fault()
                else:
                    raise ValueError(f'Invalid fault model {fault_model}')
                    
        return 0
    
    def dropout_run_faulty_campaign_on_weight(self,
                                      fault_model: str,
                                      fault_list: list,
                                      first_batch_only: bool = False,
                                      force_n: int = None,
                                      save_output: bool = False,
                                      save_ofm: bool = False,
                                      ofm_folder: str = None) -> (str, int):
        
        self.skipped_inferences = 0
        self.total_inferences = 0

        
         # Create the directory if it does not exist
        MC_output_dir = "MC_Dropout_Faulty_Results"
        if not os.path.exists(MC_output_dir):
            os.makedirs(MC_output_dir)
                    
        with torch.no_grad():
            
            if force_n is not None:
                fault_list = fault_list[:force_n]
                
            fault_list = sorted(fault_list, key=lambda x: x.injection)
            
            
            pbar2 = tqdm(fault_list,
                        colour='green',
                        desc=f'Fault Injection',
                        ncols=shutil.get_terminal_size().columns)
            
            # Create a CSV file to save the data
           
            for fault_id, fault in enumerate(pbar2):
                
        
                if fault_model == 'stuck-at_params':
                    self.__inject_fault_on_weight(fault, fault_mode='stuck-at')
                else:
                    raise ValueError(f'Invalid fault model {fault_model}')
                
                torch.cuda.reset_peak_memory_stats()
                
             
                pbar = tqdm(self.loader,
                        colour='green',
                        desc=f'fault_id {fault_id}',
                        ncols=min(120, shutil.get_terminal_size().columns))

                            
                        
                # Initialize variables for recording results            
                num_errors = 0
                pred_class = 0
                true_label = 0
                confidence = 0
                entropy = 0
                mean_logits_pred_class = 0
                var_pred_class = 0
                var_pred_class_n = 0
                all_variances = 0
                dataset_size = 0
                records = []
                    
                for batch_id, batch in enumerate(pbar):
                    
                    data, label = batch
                    data = data.to(self.device)
                    label = label.to(self.device)  
                    
                    # -------- MC Dropout multipass ------------
                    mean_logits, var_logits, mean_probs, entropy, MC_output = mc_dropout_forward(
                       self.network, data, forward_passes=20
                    )
                    # print(MC_output, 'MC_output')
                    # print(mean_logits, 'mean_logits')
                    # print(var_logits, 'var_logits')
                    preds       = mean_probs.argmax(dim=1)                # predizione finale
                    confidences = mean_probs.max(dim=1).values            # confidenza finale
                    dataset_size += label.size(0)
                    num_errors   += (preds != label).sum().item()
                    pbar.set_description(f'MC fault {fault_id}')
                    if batch_id % 10 == 0:
                        pbar.set_postfix(errors=num_errors, total=dataset_size)

                    
                    # ------ Costruzione record sample per sample --------
                    for i in range(data.size(0)):
                    
                        pred_class = preds[i].item()
                        true_label = label[i].item()
                        confidence = confidences[i].item()
                        entropy_val = entropy[i].item()
                        var_pred_class = var_logits[i, pred_class].item()
                        mean_logits_pred_class = mean_logits[i, pred_class].item()  # Classe predetta
                        # print(var_logits[i])
                        # print(var_logits[i].sum().item())
                        # print(len(var_logits[i]))
                        # exit(-1)
                        var_pred_class_n = var_logits[i, pred_class].item() / var_logits[i].sum().item()
                        # total_var = var_logits[i].sum().item()
                        all_variances = var_logits[i].tolist()  # Save all 10 variances as a list
                        # print(var_pred_class_n)
                        records.append({
                            "fault_id": fault_id,  # Add fault_id to the record
                            "pred_class": pred_class,
                            "true_label": true_label,
                            "confidence": round(confidence, 3),
                            "entropy": round(entropy_val, 9),
                            "mean logits": round(mean_logits_pred_class, 4),  # Convert to list for DataFrame
                            "var_pred_class": round(var_pred_class,5),  # Variance of the predicted class
                            "var_pred_class_n": round(var_pred_class_n,8),  # Normalized variance
                            # "total_var": round(total_var, 8),  # Total variance
                            "all_variances": [round(v, 5) for v in all_variances]
                        })

                # Costruzione finale del DataFrame
                df = pd.DataFrame(records)
                df.to_csv(f"{MC_output_dir}/fault_{fault_id}_clean_mc_dropout_predictions.csv", index=False)

                # print(f'Wrong predictions: {num_errors} / {dataset_size}')
                # print(f'Accuracy        : {100 - (100 * num_errors / dataset_size):.2f}%')
                
                # Clean the fault    
                if fault_model == 'stuck-at_params':
                    self.weight_fault_injector.restore_fault()
                else:
                    raise ValueError(f'Invalid fault model {fault_model}')
                    
        return 0
    
        
    
    def __run_inference_on_batch(self,
                                 batch_id: int,
                                 data: torch.Tensor):
        try:
            # Execute the network on the batch
            network_output = self.network(data)
            faulty_prediction = torch.topk(network_output, k=1)
            clean_prediction = torch.topk(self.clean_output[batch_id], k=1)

            # Measure the different predictions in terms of scores
            # different_predictions = int(torch.ne(faulty_prediction.values, clean_prediction.values).sum())

            # Measure the different predictions in terms of labels
            different_predictions = int(torch.ne(faulty_prediction.indices, clean_prediction.indices).sum())

            faulty_prediction_scores = network_output
            faulty_prediction_indices = [int(fault) for fault in faulty_prediction.indices]

        except RuntimeError as e:
            print(f'FaultInjectionManager: Skipped inference {self.total_inferences} in batch {batch_id}')
            print(e)
            self.skipped_inferences += 1
            faulty_prediction_scores = None
            faulty_prediction_indices = None
            different_predictions = None

        self.total_inferences += 1

        return faulty_prediction_scores, faulty_prediction_indices, different_predictions

    def __inject_fault_on_weight(self,
                                 fault,
                                 fault_mode='stuck-at') -> None:
        """
        Inject a fault in one of the weight of the network
        :param fault: The fault to inject
        :param fault_mode: Default 'stuck-at'. One of either 'stuck-at' or 'bit-flip'. Which kind of fault model to
        employ
        """

        if fault_mode == 'stuck-at':
            self.weight_fault_injector.inject_stuck_at(layer_name=f'{fault.layer_name}.weight',
                                                       tensor_index=fault.tensor_index,
                                                       bit=fault.bit,
                                                       value=fault.value)
        elif fault_mode == 'bit-flip':
            self.weight_fault_injector.inject_bit_flip(layer_name=f'{fault.layer_name}.weight',
                                                       tensor_index=fault.tensor_index,
                                                       bit=fault.bit,)
        else:
            print('FaultInjectionManager: Invalid fault mode')
            quit()


    def __inject_fault_on_neuron(self,
                                 fault: NeuronFault) -> Module:
        """
        Inject a fault in the neuron
        :param fault: The fault to inject
        :return: The injected layer
        """
        output_fault_mask = torch.zeros(size=self.injectable_modules[fault.layer_index].output_shape)

        layer = fault.layer_index
        channel = fault.feature_map_index[0]
        height = fault.feature_map_index[1]
        width = fault.feature_map_index[2]
        value = fault.value

        # Set values to one for the injected elements
        output_fault_mask[0, channel, height, width] = 1

        # Cast mask to int and move to device
        output_fault_mask = output_fault_mask.int().to(self.device)

        # Create a random output
        output_fault = torch.ones(size=self.injectable_modules[layer].output_shape, device=self.device).mul(value)

        # Inject the fault
        self.injectable_modules[layer].inject_fault(output_fault=output_fault,
                                                    output_fault_mask=output_fault_mask)

        # Return the injected layer
        return self.injectable_modules[layer]