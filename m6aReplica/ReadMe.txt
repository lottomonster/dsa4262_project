Put the Dataset0 into the folder "Dataset" and hit run all for "Data Processing Part 1.ipynb" to create a "before_embedding.parquet".


In "Data Processing Part 2.ipynb", currently using latent_dim 2/3 and use_unique_kmers = True/False. Creating total of 4 different encoders and also 4 different dataset.

In each respective folder, and their training.ipynb file, change the DATA_FILE variable name accordingly. Adjust the configuration cell accordingly and then just hit run all. Recommended to for each differnt configuration to create a new folder and put a new training.ipynb file in it. So just copy paste the "Template Folder for training". Rename the copied folder to something like the configuration used.