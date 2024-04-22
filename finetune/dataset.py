# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import typing
import aiohttp
import pandas as pd
import torch
import bittensor as bt
import wandb
from torch.utils.data import IterableDataset
import constants
import time
import numpy as np
import math
from PIL import Image
from tqdm import tqdm
from rich.console import Console
import cv2

class VisionSubsetLoader(IterableDataset):
    def __init__(self, max_samples=30000, version=6):
        self.max_samples = max_samples
        self.samples = []
        self.version = version
        asyncio.run(self.fetch())

    async def fetch(self):
        start = 0
        limit = 10000
        console = Console()
        
        with console.status("[bold cyan]Collecting datasets from vision τ subnet...", spinner_style="cyan") as status:
            while len(self.samples) < self.max_samples:
                status.update(f"[bold cyan]Collecting {len(self.samples)} datasets from vision τ subnet...")
                responses = await self.query(start, limit)
                if not responses:
                    break  # Exit the loop if no new response is obtained.
                # Add valid responses to the sample list.
                for response in responses:
                    if 'prompt' in response:
                        if f"--v {str(self.version)}" in response['prompt']:
                            response['prompt'] = self.remove_extra(response['prompt'])
                            self.samples.append(response)

                            if len(self.samples) >= self.max_samples:
                                break  # Exit as soon as the maximum number of samples is reached.

                start += limit  # Update the start for the next query

        console.print(f"[bold cyan]Finished collecting {len(self.samples)} datasets from vision τ subnet.")
    def remove_extra(self, prompt):
        return re.sub(r'\s+--[a-zA-Z0-9\-]+(\s+[^\s]+)?', '', prompt)

    async def query(self, k1=100, k2=100):
        url = 'https://app.appsmith.com/api/v1/actions/execute'
        headers = {
            'accept': 'application/json',
            'accept-language': 'en,en-US;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'content-type': 'multipart/form-data; boundary=----WebKitFormBoundarybwlWN55KMAWV3NAt',
            'origin': 'https://app.appsmith.com',
            'referer': 'https://app.appsmith.com/app/sota-images/page1-65f44e57fe8e7c034af880e3/api',
            'sec-ch-ua': '"Microsoft Edge";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0',
            'x-requested-by': 'Appsmith'
        }
        data = (b'------WebKitFormBoundarybwlWN55KMAWV3NAt\r\n'
                b'Content-Disposition: form-data; name="executeActionDTO"\r\n\r\n'
                b'{"actionId":"65f45de1650b177b352c8588","viewMode":true,"paramProperties":{"k0":{"datatype":"string","blobIdentifiers":[]},"k1":{"datatype":"number","blobIdentifiers":[]},"k2":{"datatype":"number","blobIdentifiers":[]},"k3":{"datatype":"string","blobIdentifiers":[]}},"analyticsProperties":{"isUserInitiated":false},"paginationField":"NEXT"}\r\n'
                b'------WebKitFormBoundarybwlWN55KMAWV3NAt\r\n'
                b'Content-Disposition: form-data; name="parameterMap"\r\n\r\n'
                b'{"sota_images.searchText":"k0","sota_images.pageOffset":"k1","sota_images.pageSize":"k2","sota_images.sortOrder.column ? \\"ORDER BY \\" + sota_images.sortOrder.column + \\"  \\" + (sota_images.sortOrder.order \u0021== \\"desc\\" ? \\"\\" : \\"DESC\\") : \\"\\"":"k3"}\r\n'
                b'------WebKitFormBoundarybwlWN55KMAWV3NAt\r\n'
                b'Content-Disposition: form-data; name="k0"; filename="blob"\r\n'
                b'Content-Type: text/plain\r\n\r\n\r\n'
                b'------WebKitFormBoundarybwlWN55KMAWV3NAt\r\n'
                b'Content-Disposition: form-data; name="k1"; filename="blob"\r\n'
                b'Content-Type: text/plain\r\n\r\n{0}\r\n'
                b'------WebKitFormBoundarybwlWN55KMAWV3NAt\r\n'
                b'Content-Disposition: form-data; name="k2"; filename="blob"\r\n'
                b'Content-Type: text/plain\r\n\r\n{1}\r\n'
                b'------WebKitFormBoundarybwlWN55KMAWV3NAt\r\n'
                b'Content-Disposition: form-data; name="k3"; filename="blob"\r\n'
                b'Content-Type: text/plain\r\n\r\nORDER BY timestamp  DESC\r\n'
                b'------WebKitFormBoundarybwlWN55KMAWV3NAt--\r\n')
        data_str = data.decode('utf-8')
        updated_data_str = data_str.replace("{0}", str(k1)).replace("{1}", str(k2))
        updated_data_bytes = updated_data_str.encode('utf-8')

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=updated_data_bytes) as response:
                response_data = await response.json()
                return response_data['data']['body']

    async def download_image(self, session, url, filename):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(filename, 'wb') as f:
                        f.write(await response.read())
                    return filename, 'Downloaded successfully'
                else:
                    return filename, f'Download failed with status code {response.status}'
        except Exception as e:
            return filename, f'Download failed with error: {str(e)}'
        
    def process_image(self, data_tuple, path):
        filename, prompt = data_tuple
        full_path = os.path.join(path, filename)
        new_data = []
        try:
            # Read the image using OpenCV
            img = cv2.imread(full_path)
            if img is None:
                raise IOError("Failed to read image")
            
            img_height, img_width = img.shape[:2]
            crop_size = 1024  # Define the size of each crop
            
            # Ensure the crop size does not exceed image dimensions
            crop_size = min(crop_size, img_height, img_width)
            
            # Split the image into four parts
            for i in range(4):
                if i == 0:
                    left = (i % 2) * crop_size
                    top = (i // 2) * crop_size
                    cropped_img = img[top:top+crop_size, left:left+crop_size]
                    
                    new_filename = f"{filename[:-4].replace('temp_', '')}_part_{i+1}.png"
                    new_full_path = os.path.join(path, new_filename)
                    # Save each cropped part
                    cv2.imwrite(new_full_path, cropped_img)
                    new_data.append((new_filename, prompt))
            
            # Remove the original image
            os.remove(full_path)
            
        except Exception as e:
            print(f"Failed to process image {filename}: {str(e)}")
        return new_data


    async def preprocess(self, path, erase_dir=False):
        if not os.path.exists(path):
            os.makedirs(path)
        elif erase_dir:
            os.rmdir(path)
            os.makedirs(path)
        async with aiohttp.ClientSession() as session:
            tasks = []
            data = []
            for index, item in enumerate(self.samples):
                filename = os.path.join(path, f'temp_{index+1:09d}.png')
                tasks.append(self.download_image(session, item['image_url'], filename))
                data.append((f'temp_{index+1:09d}.png', item['prompt']))

            with tqdm(total=len(tasks), desc="Downloading images") as pbar:
                async def update_progress(f):
                    await f
                    pbar.update(1)

                await asyncio.gather(*[update_progress(task) for task in tasks])

            # Splitting images into four and updating data list
            new_data = []
            with tqdm(total=len(data), desc="Splitting images") as pbar:
                with ThreadPoolExecutor() as executor:
                    results = executor.map(self.process_image, data, [path]*len(data))
                    for result in results:
                        new_data.extend(result)
                        pbar.update(1)
            # Creating a DataFrame for the CSV file
            df = pd.DataFrame(new_data, columns=['file_name', 'text'])
            df.to_csv(os.path.join(path, 'metadata.csv'), index=False)
            return new_data
    def process_sync(self, path, erase_dir=False):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.preprocess(path, erase_dir=False))
        return result
    def dump(self):
        return json.dumps(self.samples)

    def __iter__(self):
        return self.buffer.__iter__()
