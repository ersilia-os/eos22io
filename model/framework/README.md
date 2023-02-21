# Framework
## Model modifications.
Inside the Framework/code folder, all the code of the original model was added.
A folder named idl_ppb_ was created, where the original model scripts and sample input files were stored. For code optimization, organization and function calls, a module idl_pbb_modular.py was created, which contains the functions that are called from the main.py module.In the idl-ppb-original.py file the original code of the model is kept.

When carrying out a reorganization of the code, some functions that were called had their headers modified, for example the eval function now receives two parameters as input.
The imports of each module were organized to run the model, and redundant imports were eliminated.
To run the model on the cpu, all CUDA functionality was removed to work with CPU, as follows:
Added the following lines of code:
CUDA_VISIBLE_DEVICES = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
And everything that used CUDA like torch.cuda.LongTensor was changed to torch.LongTensor.
Set to install pytorch with cpu only (this is specified in the docker file).
Removed the save_models folder, since the checkpoints were stored in the checkpoints folder.
Functionalities that are not needed in ersilia were finally removed, such as painting the molecule, everything related to this functionality was removed.
For compounds like Inorganics and salts where the model is unable to calculate ppb values, the output is set to null.