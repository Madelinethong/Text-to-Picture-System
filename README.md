# Text-to-Image-System using Stable Diffusion
Overview
## Data Collection
Access the `data_processing/training_data/MAAD_Face.csv` dataset in the folder. This dataset is downloaded from https://github.com/pterhoer/MAAD-Face.

## Clean Up Dataset
1. Access the `data_processing/training_data/maad` folder for the images. The images are downloaded from https://drive.google.com/drive/folders/1ZHy7jrd6cGb2lUa4qYugXe41G_Ef9Ibw.

2. According to the `MAAD_Face.csv` file and the images located in the maad folder, there are certain file names that do not appear in both of these sources. Hence, execute `data_processing/training_data/checkCommon.py` python script to obtain the filenames that are present in both sources.

3. Next, the `MAAD_Face.csv` file which consists of approximately 3 millions rows of data is split into 3 separate csv, each contains approximately 1 million rows of data. Thus, execute `data_processing/training_data/splitFile.py` to do so.

4. Next, execute `first.py`, `second.py` and `third.py` to each generate a csv that contains the filenames that are present in the `MAAD_Face.csv` file and images located in the maad folder.

5. Next, execute `mergeCSV.py` to combine the 3 csv generated in Step 4.


As the `MAAD_Face.csv` dataset contains only male label to differentiate between male and female images, female label is added to the jsonl file.

## Data Pre-processing

## Model Training

## Example (Video)
