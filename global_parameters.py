import os

DEFAULT_DATASET_DIR = "../../datasets/"  
DEFAULT_CKPT_DIR = "../../../data/models/"   
DEFAULT_MODEL_DIR = "../../../data/pretrain_models/"  

# Map from dataset name to folder name
dataset2folder = {
    "nextgqa": "nextgqa",
    "nextqa": "nextqa"
}

# Datasets
NEXTGQA_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["nextgqa"]
)  
NEXTQA_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["nextqa"]
)  

# Models
S3D_PATH = os.path.join(
    DEFAULT_MODEL_DIR, "s3d_howto100m.pth"
)  # Path to S3D checkpoint
S3D_DICT_PATH = os.path.join(
    DEFAULT_MODEL_DIR, "s3d_dict.npy"
)  # Path to S3D dictionary
PUNCTUATOR_PATH = os.path.join(
    DEFAULT_MODEL_DIR, "INTERSPEECH-T-BRNN.pcl"
)  # Path to Punctuator2 checkpoint
TRANSFORMERS_PATH = os.path.join(
    DEFAULT_MODEL_DIR, "transformers"
)  # Path where the transformers checkpoints will be saved
