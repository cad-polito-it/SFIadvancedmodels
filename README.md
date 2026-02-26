# Fault Injection Tool for the Reliability Assessment of Deep Learning Algorithms

## Overview
**SFIadvancedmodels** is open-source software for testing the resilience of deep learning algorithms against random hardware faults. 

## Project structure

This project is organized as follows:
- `requirements.txt`: Packages to install in a virtual environment to run the application.
- `main.py`: Main entry point. Generates fault lists, runs FI campaigns (saving OFMs and outputs, both golden and faulty), and performs the final FI analysis.
- `SETTINGS.py`: Configuration file for experiment preferences.
- `utils.py`: Utility functions and helper modules.
- `faultManager/`: Files used to manage FI campaigns.
- `ofmapManager/`: Stores the golden OFMs.
- `dlModels/`: Directory where models and weights are stored.

## Setup

To get started, first clone the repository from GitHub:

`git clone https://github.com/your-username/SFIadvancedmodels.git`

## Creating a Python Environment
It is recommended to create a virtual environment to manage your dependencies. You can do this using venv:

`python3 -m venv environment_name`

`source environment_name/bin/activate`

## Installing Dependencies
Once your virtual environment is activated, install the required packages listed in requirements.txt:

`pip install -r requirements.txt`

## Usage
To generate the fault list, start a fault injection, or analyze the data, edit the `SETTINGS.py` file to configure your experiments, then run:

```bash
python3 main.py
```

Note: the injected fault is permanent and simulates a stuck-at fault in the memory that stores the model weights.

## Outputs
The code is divided into four separately activatable parts that produce different outputs, controlled by boolean variables in the `SETTINGS.py` file:

- `FAULT_LIST_GENERATION`: Generates a fault list for the selected network based on the configured parameters.
- `FAULTS_INJECTION`: Loads the fault list and executes the fault injection campaign, saving outputs and golden/corrupted OFMs according to the set preferences.
- `FI_ANALYSIS`: Compares corrupted outputs to golden outputs and reports the number of masked, non-critical, and critical (SDC-1) faults.
- `FI_ANALYSIS_SUMMARY`: Summarizes large analysis CSV files into a more accessible format when many faults are injected or large datasets are used.

The outputs produced by SFI are stored in the `output` folder. More specifically:

- `output/clean_feature_maps`: Stores the clean feature maps.
- `output/clean_output`: Stores the clean outputs.
- `output/fault_list`: The fault list used for the injections.
- `output/faulty_feature_maps`: Stores the faulty feature maps.
- `output/faulty_output`: Stores the faulty outputs.
- `results/`: Stores the analysis results.
- `results_summary/`: Stores summarized analysis results.


Files are named as follows:

- Clean FM: `batch_[batch_id]_layer_[layer_name].npz`.
	This file contains the clean output feature map of layer `[layer_name]` for input batch `[batch_id]`.
- Clean output: `clean_output.npy`.
	This file contains the clean outputs for all input batches.
- Faulty FM: `fault_[fault_id]_batch_[batch_id]_layer_[layer_name].npz`.
	This file contains the faulty output feature map of layer `[layer_name]` for input batch `[batch_id]` when fault `[fault_id]` is injected.
- Faulty output: `[fault_model]/batch_[batch_id].npy`.
	This file contains the outputs for each input batch `[batch_id]` across all injected faults.

The files are NumPy `.npy` or `.npz` arrays with the following dimensions:

- Clean FM: `B x K x H x W`
- Clean output: `N x B x C`
- Faulty FM: `B x K x H x W`
- Faulty output: `F x B x C`

Where `F` is the length of the fault list, `N` is the number of batches, `B` is the batch size, `C` is the number of classes, `K` is the number of channels in an OFM, `H` is the height of an OFM, and `W` is the width.

To load FM arrays use `np.load(file_name)['arr_0']`. To load output arrays use `np.load(file_name, allow_pickle=True)`.

## Fault list

The generated fault lists are CSV files with a specific format to which the FI refers in order to inject faults into the neural model. The structure is as follows:


Example fault list (FL) for a VGG-11 model with the GTSRB dataset

| Injection |    Layer   |   TensorIndex  | Bit |
|:---------:|:----------:|:--------------:|:---:|
|         0 | features.0 | "(3, 0, 2, 1)" |  15 |
|    ...    |     ...    |       ...      | ... |

- `Injection`: Injection number.
- `Layer`: The layer in which the fault is injected.
- `TensorIndex`: Coordinates of the weight tensor where the fault is injected.
- `Bit`: The corrupted bit that is flipped.


## Analyses

The analysis files produced by the `FI_ANALYSIS` option are stored in the `results/` folder and are organized by dataset, model, and batch size: `results/dataset-name/model-name/batch-size/`.
Inside that folder there are two files:

- `fault_statistics.txt`: Text file containing the total counts of masked, non-critical, and critical (SDC-1) inferences.
- `output_analysis.csv`: CSV file containing classification details for every fault and inference.

Faults are classified into three categories:
- `masked`: Inference that masks the fault.
- `non-critical`: Inference where the fault alters outputs but does not change the predicted class.
- `critical (SDC-1)`: Inference classified as SDC-1, meaning it changes the final prediction.



The `output_analysis.csv` is organized as follows:

| fault | batch | image | output |
|:-----:|:-----:|:-----:|:------:|
|     0 |     0 |     0 |      1 |
|     0 |     0 |     1 |      0 |
|     0 |     0 |     2 |      0 |
|     0 |     0 |     3 |      2 |
|  ...  |  ...  |  ...  |   ...  |
| 16663 |     9 |  1024 |      1 |

- `fault`: Unique identifier of the injected fault, corresponding to the `Injection` column in the fault list used.
- `batch`: Batch index containing the dataset images used for inference.
- `image`: Index of the image in the batch on which the inference was performed.
- `output`: Classification of the injected fault by comparing golden outputs with corrupted outputs. Values: `0` = masked, `1` = non-critical, `2` = critical (SDC-1).

## Summarized analysis

When many faults are injected or a large dataset is used, `output_analysis.csv` can become large and hard to read. Using the `FI_ANALYSIS_SUMMARY` option generates a summary CSV named `model-name_summary.csv` inside `results_summary/dataset-name/model-name/batch-size/`. This file combines the original fault list with summarized results for each fault. The CSV is organized as follows:

| Injection | Layer |   TensorIndex   | Bit | n_injections | masked | non_critical | critical |
|:---------:|:-----:|:---------------:|:---:|:------------:|:------:|:------------:|:--------:|
|         0 | conv1 |  "(7, 0, 2, 1)" |  15 |        10000 |  10000 |            0 |        0 |
|         1 | conv1 | "(14, 0, 2, 0)" |   5 |        10000 |  10000 |            0 |        0 |
|         2 | conv1 | "(27, 0, 0, 0)" |  13 |        10000 |    701 |         9298 |        1 |
|         3 | conv1 | "(14, 2, 2, 0)" |  12 |        10000 |   9998 |            2 |        0 |
|    ...    |  ...  |       ...       | ... |      ...     |   ...  |      ...     |    ...   |

- `Injection`: Injection number.
- `Layer`: The layer in which the fault is injected.
- `TensorIndex`: Coordinates of the weight tensor where the fault is injected.
- `Bit`: The corrupted bit that is flipped.
- `n_injections`: Number of inferences performed with the injected fault (i.e., number of dataset examples executed).
- `masked`: Number of inferences classified as masked.
- `non_critical`: Number of inferences classified as non-critical.
- `critical`: Number of inferences classified as critical (SDC-1).

## Acknowledgments

This study was carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-GenerationEU (PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) – MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.3 – D.D. 1555 11/10/2022, PE00000013). This manuscript reflects only the authors’ views and opinions; neither the European Union nor the European Commission can be held responsible for them.

## Main Contributors
- Annachiara Ruospo (annachiara.ruospo@polito.it)
- Vittorio Turco (vittorio.turco@polito.it)
- Gabriele Gavarini

