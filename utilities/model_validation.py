import asyncio
import sys
import math
import random

from model.data import ModelId
from finetune.dataset import VisionSubsetLoader
from finetune.validation import compute_losses
from model.data import Model
import argparse
import constants
import torch
from model.model_updater import ModelUpdater
from utilities.perf_monitor import PerfMonitor
from transformers import CLIPTextModelWithProjection, CLIPTextModel
from diffusers import DDPMScheduler
def load_model(model_path, parameters: constants.CompetitionParameters):
    model_id = ModelId(namespace="namespace", name=model_path, competition_id=parameters.competition_id)
    unet_model = parameters.architecture.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=False,
        subfolder="unet",
        use_safetensors=True,
        **parameters.kwargs
    )
    return Model(id=model_id, unet=unet_model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, help="Local path to your model", required=True
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device name.",
    )
    parser.add_argument(
        "--latest_vision_samples",
        type=int,
        default=65,
        help="Number of most recent Vision samples to eval against",
    )
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Implementation of attention to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="datatype to load model in, either bfloat16 or float16",
    )
    parser.add_argument(
        "--competition_id",
        type=str,
        default=constants.ORIGINAL_COMPETITION_ID,
        help="competition to validate against (use --list-competitions to get all competitions)"
    )
    parser.add_argument(
        "--list_competitions",
        action="store_true",
        help="Print out all competitions"
    )
    args = parser.parse_args()
    if args.list_competitions:
        print(constants.COMPETITION_SCHEDULE)
        return
    
    competition_parameters = ModelUpdater.get_competition_parameters(args.competition_id)
    competition_parameters.kwargs["torch_dtype"] = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"Loading model for competition {args.competition_id}")
    load_model_perf = PerfMonitor("Eval: Load model")
    with load_model_perf.sample():
        model = load_model(args.model_path, competition_parameters)
    print(load_model_perf.summary_str())
    if not ModelUpdater.verify_model_satisfies_parameters(model):
        print("Model does not satisfy competition parameters!!!")
        return
    del model.unet
    
    sota_perf = PerfMonitor("Downloading: Vision Dataset")
    print(f"Downloading Vision datasets..")
    loader = VisionSubsetLoader(max_samples=100, version=6)
    loader.process_sync(path=constants.CACHE_DIR_DATASET, erase_dir=True)
    sota_perf.summary_str()
    print("Calculating losses")
    compute_loss_perf = PerfMonitor("Eval: Compute loss")
    with compute_loss_perf.sample():
        losses = compute_losses(model_id=model.id.name, dataset_folder=constants.CACHE_DIR_DATASET)
    print(compute_loss_perf.summary_str())

    average_model_loss = sum(losses) / len(losses) if len(losses) > 0 else math.inf

    print(f"The average model loss for {args.model_path} is {average_model_loss}")


if __name__ == "__main__":
    main()
