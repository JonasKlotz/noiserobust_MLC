This is the technical documentation of our project with [RSIM of TU BERLIN](https://rsim.berlin/).

# Setup
First install the conda environment at `cluster_instructions/conda_env.yml` and activate it.

## Data Set (Deep Globe Patches)
Then download the Deepglobe dataset from [KAGGLE](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset). 
Place it in the project with the path `data/deepglobe`. Then run the patch-sampling pipeline: `data_pipeline/deepglobe/patch_sampling.py`. This creates LMDB files for the train, test and valid set at `data/deepglobe_patches/[train/test/valid]/`.

## Word Embeddings
 The folder `data/glove` contains the embeddings for the deepglobe-labels for embeddings spaces of 50 and 300. If other embeddings are necessary, or a different dataset is used. These can be created by modifying the `src/wordembedding/glove.py` and running it. To use this downlaod the glove txt files from [Stanford](https://nlp.stanford.edu/projects/glove/). Currently d = [50,100,200,300] are available there. For different embedding size, retrain the glove model.


## Run Example
The most important parameters are contained in this exemplary run.
For all parameters check out `src/config_args.py`, for our parameterized runs check out `cluster_instructions/<model>_<loss>.sh`

`python main.py -model CbMLC -loss weighted_bce -optim sgd -d_model 50-lr 0.0001 -add_noise 0.1 -sub_noise 0.1`

