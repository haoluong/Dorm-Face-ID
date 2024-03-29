# initialize Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
 
# initialize constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"
 
# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

#initialize ultilities
CHECKPOINT_PATH = "/home/hao/DCLV-HK191/experiment_on_ktx/ktx-checkpoint/cp.ckpt"
