BATCH_SIZE = 32

BARS_COUNT = 50
RESET_ON_CLOSE = True
RANDOM_OFS_ON_RESET = True
REWARD_ON_CLOSE = True
STATE_1D = True

TARGET_NET_SYNC = 1000

GAMMA = 0.99

REPLAY_SIZE = 100_000
REPLAY_INITIAL = 10_000

REWARD_STEPS = 2

LEARNING_RATE = 0.0001

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 1_000_000

CHECKPOINT_EVERY_STEP = 1_000_000
VALIDATION_EVERY_STEP = 100_000

default_data_paths = {
    "BTCUSD": "./data/1M/BTCUSD/1.csv",
    "BTCEUR": "./data/1M/BTCEUR/1.csv",
    "BCHUSD": "./data/1M/BCHUSD/1.csv",
    "EOSUSD": "./data/1M/EOSUSD/1.csv",
    "LTCUSD": "./data/1M/LTCUSD/1.csv",
    "XMRUSD": "./data/1M/XMRUSD/1.csv",
    "XRPEUR": "./data/1M/XRPEUR/1.csv",
    "XRPUSD": "./data/1M/XRPUSD/1.csv",
    "XTZUSD": "./data/1M/XTZUSD/1.csv",
}

default_validation_paths = {
    "ETHEUR": "./data/1M/ETHEUR/1.csv",
    "ETHUSD": "./data/1M/ETHUSD/1.csv",
}
