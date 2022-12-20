# NIAT

For MOSI and MOSEI dataset, you need to download them Using the following link. (aligned_50.pkl).
And Change the root_dataset_dir in __datasetCommonParams in config/config_regression.py.


- [BaiduYun Disk](https://pan.baidu.com/s/1XmobKHUqnXciAm7hfnj2gg) `code: mfet`
- [Google Drive](https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk?usp=sharing)

For experiment of the asr error and sentiment word replacement, you should download the corresponding data Here.

- [Google Drive](https://drive.google.com/drive/folders/1L7wsUTk5spP_hJRNqTS5sfbOX_40BxwS)

Using the following script to run our codes.

```python
python run.py --modelName niat --datasetName mosi
```

and evaluate the trained model with others type of modality feature missing.

For performance reproduce, we provided the saved model parameters. You should unzip the niat-{mosi/mosei}-method_one-0.2-{111x}.pth file from [Here](https://drive.google.com/drive/folders/1PWL47lviWhFBhklg1HGBEm3APbabmWDh), and place them into saved_models/normals then run the Test.py.


```python
python Test.py --modelName niat --datasetName mosi --noise_type [test_noise_type]
```

