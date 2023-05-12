# Text-to-Image-System using Stable Diffusion
Overview
## Data Collection
Access the `data_processing/training_data/MAAD_Face.csv` dataset in the folder. This dataset is downloaded from https://github.com/pterhoer/MAAD-Face.

## Clean Up Dataset
1. Access the `data_processing/training_data/maad` folder for the images. The images are downloaded from https://drive.google.com/drive/folders/1ZHy7jrd6cGb2lUa4qYugXe41G_Ef9Ibw.

2. According to the `MAAD_Face.csv` file and the images located in the maad folder, there are certain file names that do not appear in both of these sources. Hence, execute `data_processing/training_data/checkCommon.py` python script to obtain the filenames that are present in both sources.

3. The `MAAD_Face.csv` file which consists of approximately 3 millions rows of data is split into 3 separate csv, each contains approximately 1 million rows of data. Thus, execute `data_processing/training_data/splitFile.py` python script to do so.

4. Execute `data_processing/training_data/first/first.py`, `data_processing/training_data/second/second.py` and `data_processing/training_data/third/third.py` python scripts to each generate a csv that contains the filenames that are present in the `MAAD_Face.csv` file and images located in the maad folder.

5. Execute `data_processing/training_data/mergeCSV.py` python script to combine the 3 csv generated in Step 4.

6. Execute `data_processing/training_data/csvToJsonl.py` python script to convert the csv generated in Step 5 to a jsonl file. Originally, the MAAD-Face dataset consists of 47 attributes/labels. However, after refining the dataset to retain only the useful attributes, the generated jsonl file comprises 32 useful attributes.

(ADD THE TABLE)

## Data Pre-processing
Execute `data_processing/training_data/metadata-maad-full-version.py` python script to pre-process the data.

The steps taken for pre-processing of the data are as such:
1. Due to the `MAAD_Face.csv` dataset solely comprising a male label to distinguish between male and female images, a female label is added to the dataset.
2. Since there is a typographical error in the spelling of the `blond hair` attribute in the `MAAD_Face.csv` dataset, it is corrected to `blonde hair`.
3. To train the model, a set of 100,000 data is selected randomly. Access the `data_processing/training_data/maad_face_lite_version/metadata.jsonl` for the jsonl file that contains these 100,000 data.

Finally, execute `data_processing/training_data/changeFileName.py` python script to edit and update the file path name in `metadata.jsonl`.

## Model Training
### Setup 
For this project, the programming languages and development environment used are:
1. Programming languages - Python, HTML, CSS
2. Development environment: Google Colab

In Google Colab, run the following command to access Google Drive from Google Colab:

    from google.colab import drive
    drive.mount('/content/drive')

### Python Packages
Activate the virtual environment and run the following command to install the dependencies:

    pip install accelerate torchvision transformers datasets ftfy tensorboard

Next, install the diffusers package as follows:

    pip install diffusers

For the latest development version of diffusers, kindly install it using the following command:
  
    pip install git+https://github.com/huggingface/diffusers

### Acelerate
To configure `accelerate`, prepare a new file called `config.yaml` and append the following content:

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: NO
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
use_cpu: false
```

During training, simply specify the `config_file` argument and point it to the path of the config file:

```python
accelerate launch --config_file config.yaml train_text_to_image_lora.py ...
```





## Example (Video)
