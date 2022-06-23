import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
NUM_EPOCHS_PRE = 50
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "./evaluations/_disc.pth.tar"
CHECKPOINT_DISC_PRE = "./output/_disc.pth.tar"
CHECKPOINT_GEN = "./evaluations/_gen.pth.tar"
CHECKPOINT_GEN_PRE = "./output/_gen.pth.tar"
