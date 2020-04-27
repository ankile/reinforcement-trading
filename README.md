# Deep Q-Learning for Trading Cryptocurrencies

### Running in colab

Just open one of the training notebooks or the testing notebook and press the "Open in Colab"-button at the top to launch the code in Colab.

### Running locally

Clone this repo to local computer. `cd` into the repo-folder.

Create virtualenv

`virtualenv .env -p <path to python binary to use>` e.g. `virtualenv .env -p python3`

Enter virtualenv

`source .env/bin/activate`

Install requirements

`pip install -r requirements.txt --no-deps`

Run the training file (and/or do desired changes before starting training)

`python train_agent.py`

### Notes

The testing file is only available in a colab format
