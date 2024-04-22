import asyncio
import os
from pathlib import Path
import sys
import click
import json
from rich.console import Console
from rich.prompt import IntPrompt, Confirm, Prompt
@click.group()
def model():
    """Model commands."""
    pass

@model.command(context_settings={"ignore_unknown_options": True})
@click.argument('model_type', type=click.Choice(['sdxl'], case_sensitive=False))
@click.argument('extra_args', nargs=-1)  # Capture all additional arguments
def train(model_type, extra_args):
    import scripts.train_diffusion_sdxl as train_sdxl
    extra_args_str = ' '.join(extra_args)
    # Backup original sys.argv
    original_argv = sys.argv.copy()
    
    # Set sys.argv to simulate command line arguments for the module
    sys.argv = ['pixel model train'] + list(extra_args) # 'script_name' is a placeholder for argv[0]
    if model_type == "sdxl":
        train_sdxl.exec()
