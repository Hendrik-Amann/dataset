# dataset
Format ArXiv dataset and split into train, val, test

Used package versions on ubuntu:
- Python 3.10.2
- Spark 3.2.0
- openjdk 11.0.20.1
Python package's version:
- PySpark 3.2.0
- Transformers transformers 4.34.0.dev0
- Pandas 2.1.0

1. run tokenizeDataFile.py with val and test file as input; cleans Section Names, counts tokens with pegasus and led tokenizer for article texts; returns single file with 2 more columns (token counts)
2. run filter with file from step 1 as input; filters for <= 16384 tokens and filters for at least 1 match for dancer, top 500 and shuffle; returns single file
3. run split with file from step 2, create train, val, test files
4. run basicdataset with files from step 3; drops some unncessary columns and concatenates; generate train, val, test files for basictest
