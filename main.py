import copy
import os
from pyexpat import model
import torch
import SETTINGS
from faultManager.FaultListManager import FLManager
from faultManager.FaultInjectionManager import FaultInjectionManager
from ofmapManager.OutputFeatureMapsManager import OutputFeatureMapsManager
from utils import get_network, get_device, get_loader, get_fault_list, clean_inference, output_definition,  \
                  get_fault_list, clean_inference, output_definition,  fault_list_gen, csv_summary, \
                  image_segmentation_clean_inference, segmentation_clean_output,csv_summary_segmentation
   
from utils_MC_dropout import clean_dropout_MC_inference, set_dropout_probability, dropout_layers_activation, check_dropout_and_batchnorm_status


def main():

    if SETTINGS.FAULT_LIST_GENERATION:
        
        fault_list_gen()
        
    else:
        
        print('Fault list generation is disabled')
    
    if SETTINGS.FAULTS_INJECTION or SETTINGS.ONLY_CLEAN_INFERENCE:
        # Set deterministic algorithms
        
        if SETTINGS.IMAGE_SEGMENTATION:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Oppure ":16:8"
            torch.use_deterministic_algorithms(mode=True)
        else:
            torch.use_deterministic_algorithms(mode=True)


        # Select the device
        device = get_device(use_cuda0=SETTINGS.USE_CUDA_0,
                            use_cuda1=SETTINGS.USE_CUDA_1)
        
        print(f'Using device {device}')
         
         
        # Load the network
        network = get_network(network_name=SETTINGS.NETWORK,
                            device=device,
                            dataset_name=SETTINGS.DATASET)
        
        
        # Load the dataset
        loader = get_loader(network_name=SETTINGS.NETWORK,
                            batch_size=SETTINGS.BATCH_SIZE,
                            dataset_name=SETTINGS.DATASET)
        
        
        
        if SETTINGS.ONLY_CLEAN_INFERENCE: 
            
            if SETTINGS.IMAGE_CLASSIFICATION:
                print('clean inference accuracy test:')
                clean_inference(network, loader, device, SETTINGS.NETWORK)
                exit(-1)
            elif SETTINGS.IMAGE_SEGMENTATION:
                print('clean inference accuracy test:')
                image_segmentation_clean_inference(network, loader, device, network_name=SETTINGS.NETWORK)
                exit(-1)
            else:
                raise ValueError("Unsupported task type. Please check SETTINGS configuration.")
        

        # Folder containing the feature maps
        clean_fm_folder = SETTINGS.CLEAN_FM_FOLDER
        faulty_fm_folder = SETTINGS.FAULTY_FM_FOLDER
        
        os.makedirs(clean_fm_folder, exist_ok=True)
        os.makedirs(faulty_fm_folder, exist_ok=True)

        # Folder containing the clean output
        clean_output_folder = SETTINGS.CLEAN_OUTPUT_FOLDER

        #attenzione a module_classes che mi salva ofm diverse!
        module_classes = SETTINGS.MODULE_CLASSES
        
        feature_maps_layer_names = [name.replace('.weight', '') for name, module in network.named_modules()
                                            if isinstance(module, module_classes)]
        
        print('feature maps layer names:')
        print(feature_maps_layer_names)
    
        clean_ofm_manager = OutputFeatureMapsManager(network=network,
                                                    loader=loader,
                                                    module_classes=SETTINGS.MODULE_CLASSES,
                                                    device=device,
                                                    fm_folder=clean_fm_folder,
                                                    clean_output_folder=clean_output_folder)

        # Try to load the clean input
        if SETTINGS.IMAGE_CLASSIFICATION:
            clean_ofm_manager.load_clean_output()
        elif SETTINGS.IMAGE_SEGMENTATION:
            segmentation_clean_output(device=device, model=network, dataloader=loader)
        else:
            raise ValueError("Unsupported task type. Please check SETTINGS configuration.")

        
        
        # if  SETTINGS.IMAGE_CLASSIFICATION and SETTINGS.MC_DROPOUT:
        #     print('clean inference accuracy test with MC Dropout:')
        #     set_dropout_probability(model=network, dropout_probability=SETTINGS.DROPOUT_PROBABILITY)
        #     dropout_layers_activation(model=network)
        #     check_dropout_and_batchnorm_status(model=network)
        #     clean_dropout_MC_inference(network, loader, device, SETTINGS.BATCH_SIZE)
            
        # Generate fault list
        fault_list_generator = FLManager(network=network,
                                                network_name=SETTINGS.NETWORK,
                                                device=device,
                                                module_class=SETTINGS.MODULE_CLASSES_FAULT_LIST,
                                                input_size=loader.dataset[0][0].unsqueeze(0).shape,
                                                save_ifm=True)

        # Create a smart network. a copy of the network with its convolutional layers replaced by their smart counterpart
        # smart_network = copy.deepcopy(network)
        # fault_list_generator.update_network(network)

        # Manage the fault models
        
        fault_list, injectable_modules = get_fault_list(fault_model=SETTINGS.FAULT_MODEL,
                                                        fault_list_generator=fault_list_generator)

        # Execute the fault injection campaign with the smart network
        fault_injection_executor = FaultInjectionManager(network=network,
                                                        network_name=SETTINGS.NETWORK,
                                                        device=device,
                                                        loader=loader,
                                                        clean_output=clean_ofm_manager.clean_output,
                                                        injectable_modules=injectable_modules)
        
        if SETTINGS.IMAGE_CLASSIFICATION and SETTINGS.MC_DROPOUT == False:
            fault_injection_executor.run_faulty_campaign_on_weight(fault_model=SETTINGS.FAULT_MODEL,
                                                                fault_list=fault_list,
                                                                first_batch_only=False,
                                                                force_n=SETTINGS.FAULTS_TO_INJECT,
                                                                save_output=SETTINGS.SAVE_FAULTY_OUTPUT,
                                                                save_ofm=SETTINGS.SAVE_FAULTY_OFM,
                                                                ofm_folder=faulty_fm_folder)
        elif SETTINGS.IMAGE_CLASSIFICATION and SETTINGS.MC_DROPOUT:
            set_dropout_probability(model=network, dropout_probability=SETTINGS.DROPOUT_PROBABILITY)
            dropout_layers_activation(model=network)
            check_dropout_and_batchnorm_status(model=network)
            fault_injection_executor.dropout_run_faulty_campaign_on_weight(fault_model=SETTINGS.FAULT_MODEL,
                                                                fault_list=fault_list,
                                                                first_batch_only=False,
                                                                force_n=SETTINGS.FAULTS_TO_INJECT,
                                                                save_output=SETTINGS.SAVE_FAULTY_OUTPUT,
                                                                save_ofm=SETTINGS.SAVE_FAULTY_OFM,
                                                                ofm_folder=faulty_fm_folder)
        elif SETTINGS.IMAGE_SEGMENTATION:
            fault_injection_executor.run_faulty_campaign_on_weight_segmentation(fault_model=SETTINGS.FAULT_MODEL,
                                                            fault_list=fault_list,
                                                            first_batch_only=False,
                                                            force_n=SETTINGS.FAULTS_TO_INJECT,
                                                            save_output=SETTINGS.SAVE_FAULTY_OUTPUT,
                                                            save_ofm=SETTINGS.SAVE_FAULTY_OFM,
                                                            ofm_folder=faulty_fm_folder)
        else:
            raise ValueError("Unsupported task type. Please check SETTINGS configuration")
            
        
    else:
        print('Fault injection is disabled')
        
    if SETTINGS.FI_ANALYSIS and SETTINGS.IMAGE_CLASSIFICATION:
        try:
            output_definition(test_loader=loader, batch_size=SETTINGS.BATCH_SIZE)
            print('Done')
        except:
            print('No loader found to save the labels, creating a new one')
            _, loader = get_loader(network_name=SETTINGS.NETWORK,
                            batch_size=SETTINGS.BATCH_SIZE,
                            dataset_name=SETTINGS.DATASET)
            output_definition(test_loader=loader, batch_size=SETTINGS.BATCH_SIZE)
            print('Done')
            
       
        
        
    else:
        print('Fault injection analysis is disabled')
    
    if SETTINGS.FI_ANALYSIS_SUMMARY:
        if SETTINGS.IMAGE_CLASSIFICATION:
            print('Generating csv summary')
            csv_summary()
            print('csv summary generated')
        elif SETTINGS.IMAGE_SEGMENTATION:
            print('Generating csv summary')
            csv_summary_segmentation()
            print('csv summary generated')
        else:
            raise ValueError("Unsupported task type. Please check SETTINGS configuration")

        



if __name__ == '__main__':
    main()