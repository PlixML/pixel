from pathlib import Path
from dataclasses import dataclass
from transformers import PreTrainedModel, LlamaForCausalLM, GemmaForCausalLM, StableLmForCausalLM
from typing import Type, Optional, Any, List, Tuple
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
import math

@dataclass
class CompetitionParameters:
    """Class defining model parameters"""

    # The maximum parameter size allowed for models
    max_model_parameter_size: int
    # Architecture class of model
    architecture: Type[PreTrainedModel]

    autoencoder: Any
    # Any additional arguments to from_pretrained
    kwargs: Any
    # Fixed tokenizer
    base_model: Optional[str]

    vae_model: Optional[str]

    vae_path: Optional[str]
    # Reward percentage
    reward_percentage: float
    # Competition id
    competition_id: str
    # Dataset id
    dataset: str

# ---------------------------------
# Project Constants.
# ---------------------------------

# The validator WANDB project.
WANDB_PROJECT = "pixel-subnet"
# The uid for this subnet.
SUBNET_UID = 17
# The start block of this subnet
SUBNET_START_BLOCK = 2225782
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo
MAX_HUGGING_FACE_BYTES: int = 6 * 1024 * 1024 * 1024
# Schedule of model architectures
COMPETITION_SCHEDULE: List[CompetitionParameters] = [
    CompetitionParameters(
        max_model_parameter_size=2.5 * 1024 * 1024 * 1024,
        architecture=UNet2DConditionModel,
        autoencoder=AutoencoderKL,
        kwargs={},
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        vae_path="vae",
        dataset="midjourney_v6_all",
        reward_percentage=1.0,
        competition_id="m6"
    ),
]
ORIGINAL_COMPETITION_ID = "m6"


assert math.isclose(sum(x.reward_percentage for x in COMPETITION_SCHEDULE), 1.0)
assert all(len(x.competition_id) > 0 and len(x.competition_id) <= 5 for x in COMPETITION_SCHEDULE)

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = 2002

# validator weight moving average term
alpha = 0.9
# validator scoring exponential temperature
temperature = 0.08
# validator score boosting for earlier models.
timestamp_epsilon = 0.0005
# validator eval sequence length.
sequence_length = 2048

# norm validation values
norm_eps_soft = 500
norm_eps_soft_percent_threshold = 0.3
norm_eps_hard = 2500
