# Miner

Miners train locally and periodically publish their best model to 🤗 Hugging Face and commit the metadata for that model to the Bittensor chain.

Miners can only have one model associated with them on the chain for evaluation by validators at a time.

The communication between a miner and a validator happens asynchronously chain and therefore Miners do not need to be running continuously. Validators will use whichever metadata was most recently published by the miner to know which model to download from 🤗 Hugging Face.

# System Requirements

Miners will need enough disk space to store their model as they work on. Each uploaded model (As of Jan 1st, 2024) may not be more than 15 GB. It is recommended to have at least 25 GB of disk space.

Miners will need enough processing power to train their model. The device the model is trained on is recommended to be a large GPU with atleast 48 GB of VRAM.

# Getting started

## Prerequisites

1. Get a Hugging Face Account: 

Miner and validators use 🤗 Hugging Face in order to share model state information. Miners will be uploading to 🤗 Hugging Face and therefore must attain a account from [🤗 Hugging Face](https://huggingface.co/) along with a user access token which can be found by following the instructions [here](https://huggingface.co/docs/hub/security-tokens).

Make sure that any repo you create for uploading is public so that the validators can download from it for evaluation.

2. Clone the repo

```shell
git clone https://github.com/PlixML/pixel.git
```

3. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

4. Install the requirements. From your virtual environment, run
```shell
cd pixel
python -m pip install -e .
```

5. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

6. (Optional) Run a Subtensor instance:

Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's. 
We recommend running a local node as follows and passing the ```--subtensor.network local``` flag to your running miners/validators. 
To install and run a local subtensor node follow the commands below with Docker and Docker-Compose previously installed.
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker compose up --detach
```
---

# Running the Miner

There is no script to run a miner. You must train a model using `pixel dataset generate` and `pixel model train sdxl`.

See [Validator Psuedocode](validator.md#validator) for more information on how they the evaluation occurs.

## Env File

The Miner requires a .env file with your 🤗 Hugging Face access token in order to upload models.

Create a `.env` file in the `pixel` directory and add the following to it:
```shell
HF_ACCESS_TOKEN="YOUR_HF_ACCESS_TOKEN"
```

## Generate a dataset

To start your miner the most basic command is

```shell
pixel dataset generate
```

- `Number of samples`: Number of 2048x2048 x4 images to download from subnet 19

- `Midjourney Version`: Midjourney dataset version.

- `Preprocessing the images`: Create a folder ready for training with pixel model train

---
## Train an SDXL model

To start your miner the most basic command is

```shell
accelerate launch scripts/train_diffusion_sdxl.py \
            --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
            --train_data_dir preprocessed_images \
            --image_column image \
            --caption_column text \
            --learning_rate 2e-6 \
            --noise_offset 0.1 \
            --snr_gamma 5 \
            --train_batch_size 1 \
            --num_epochs 10

```

- `pretrained_model_name_or_path`: SDXL Model (Take the best model of the subnet)

- `train_data_dir`: Folder where you generated the datasets

- `image_column`: image column in metadata.csv file

- `caption_column`: caption column in metadata.csv file

- `learning_rate`: Learning rate

- `noise_offset`: Noise_offset https://www.crosslabs.org//blog/diffusion-with-offset-noise

- `snr_gamma`: Min-SNR value [caption column in metadata.csv file](https://huggingface.co/papers/2303.09556)

- `train_batch_size`: Train Batch Size. Increase it if your GPU supports it

---

## Manually uploading a model

In some cases you may have failed to upload a model or wish to upload a model without further training.

Due to rate limiting by the Bittensor chain you may only upload a model every 20 minutes.

You can manually upload with the following command:
```shell
python scripts/upload_model.py --load_model_dir <path to model> --hf_repo_id my-username/my-project --wallet.name coldkey --wallet.hotkey hotkey --competition_id m6
```

## Running a custom Miner

As of April 1st, 2024 the subnet works with SDXL models supported by constraints:
1. Has less than 2.5B parameters.
2. Total size of the repo is less than 6 Gigabytes.
The `finetune/mining.py` file has several methods that you may find useful. Example below.

```python
import pretrainas ft
import bittensor as bt
from transformers import PreTrainedModel

config = bt.config(...)
wallet = bt.wallet()
metagraph = bt.metagraph(netuid=6)

actions = ft.mining.actions.Actions.create(config, wallet)

# Load a model from another miner.
model: PreTrainedModel = actions.load_remote_model(uid=123, metagraph=metagraph, download_dir="mydir")

# Save the model to local file.
actions.save(model, "model-foo/")

# Load the model from disk.
actions.load_local_model("model-foo/")

# Publish the model for validator evaluation.
actions.push(model)
```
