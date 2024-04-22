import click

from .dataset import dataset
from .model import model
def create_pixel_cli() -> click.Group:
    # Initialize Rich console
    @click.group()
    def pixel():
        """pixel CLI."""
        pass
    pixel.add_command(dataset)
    pixel.add_command(model)
    return pixel
def main():
    cli = create_pixel_cli()
    cli()
if __name__ == '__main__':
    main()