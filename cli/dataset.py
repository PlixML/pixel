import asyncio
import os
from pathlib import Path
import click
import json
from rich.console import Console
from rich.prompt import IntPrompt, Confirm, Prompt
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
@click.group()
def dataset():
    """Dataset commands."""
    pass

@dataset.command()

def generate():
    from finetune.dataset import VisionSubsetLoader
    loop = asyncio.get_event_loop()
    console = Console()
    console.print('[cyan bold]Pixel ・ Midjourney Dataset Generator from vision τ subnet')
    
    num_samples = IntPrompt.ask("Please enter the number of samples you want to generate", default=10000)
    version = IntPrompt.ask("Please enter the version of MidJourney you want to use", default=6)
    
    loader = VisionSubsetLoader(max_samples=num_samples, version=version)
    
    if Confirm.ask("Do you want to preprocess the images?"):
        path = Prompt.ask("Please enter the path where the images should be preprocessed", default="./preprocessed_images")
        os.makedirs(path, exist_ok=True)
        processing = loader.process_sync(path=path)
        console.print(f"[green bold]Processing complete! Number of images processed: {len(processing)} / Dir = {path}")

        # Define the command with proper line breaks and spacing
        command_text = """[bold blue]To train model with 1 GPU, use \npixel model train sdxl \\
            --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \\
            --train_data_dir preprocessed_images \\
            --image_column image \\
            --caption_column text \\
            --train_batch_size 1 """
        command_text_mg = """[bold blue]To train model with multiple GPUs, generate a config with accelerate config and use \naccelerate launch scripts/train_diffusion_sdxl.py \\
            --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \\
            --train_data_dir preprocessed_images \\
            --image_column image \\
            --caption_column text"""
        # Print the panel containing the command
        console.print(command_text)
        console.print(command_text_mg)
    else:
        dataset_name = Prompt.ask("Please enter the name of the dataset file you wish to create", default="dataset.json")
        dataset_json = loader.dump()
        
        with open(dataset_name, 'w') as file:
            file.write(dataset_json)
            console.print(f"Dataset '{dataset_name}' created.", style="green bold")
