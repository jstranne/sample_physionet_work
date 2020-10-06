# sample_physionet_work
How to run the code
## Proprocessing Data
To create the list for preprocessing. Run the list_training_files.py in the main folder. This will create the "training_names.txt" file that lists all of the names of the training files.

Next, use the CreateRPDataset.py file in order to window and preprocess the data. This will create files like the "tr03-0078_windowed_preprocess.npy" files in the training folder.

## Testing RP data
