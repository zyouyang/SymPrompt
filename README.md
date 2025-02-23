# Symbolic Prompt Tuning Completes the App Promotion Graph

Welcome to the official code repository for the **ECML PKDD 2024 ADS** paper "Symbolic Prompt Tuning Completes the App Promotion Graph". The data can be accessed through [this link](https://www.dropbox.com/scl/fi/bb4eo2yiekwfrecsvsj35/my_data.zip?rlkey=vehg3lj2oabb8yo0h19db1w0l&st=vhfhb6w2&dl=0). Please download it and place it under the main folder.

The paper can be accessed through [this link]([https://www.dropbox.com/scl/fi/bb4eo2yiekwfrecsvsj35/my_data.zip?rlkey=vehg3lj2oabb8yo0h19db1w0l&st=vhfhb6w2&dl=0](https://zyouyang.github.io/assets/publications/SymPrompt.pdf))

## Code Structure
- `main.py`

  Running the model on the data stored in my_data folder.
  
- `parser_hsf.py`
  
  Setting the arguments for the running.

- `prompts/`

  Containing codes that generate prompts as needed.
  - `gen_metarules.py` generates rules according to metapaths for each query. The output file is `all_pathcnt_2.pkl`.
  - `gen_prompts.py` generates the related prompts. It can be set to metapath-only, embedding-based only, or both. This part relies on [this repo](https://github.com/salesforce/MultiHopKG) for embedding-based prompts generation. To run it, please clone the above repo under the main folder, and set the paths accordingly. We have provided the pre-processed data in [this link](https://www.dropbox.com/scl/fi/bb4eo2yiekwfrecsvsj35/my_data.zip?rlkey=vehg3lj2oabb8yo0h19db1w0l&st=vhfhb6w2&dl=0) if the user would like to skip this step.
  
## How to Run
Under the main directory, run `python main.py --rand_format`.
