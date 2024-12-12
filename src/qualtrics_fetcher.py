import asyncio
import csv
import io
import os
import re
import wave
import zipfile
from asyncio import Semaphore
from os import path
from typing import List, Optional

import aiohttp

headers = {"X-API-TOKEN": os.environ["qualtrics_key"]}

# Add rate limit configuration
RATE_LIMIT = 5  # requests per second
MAX_CONCURRENT = 3  # maximum concurrent requests


def concat_wavs(wav_contents: List[bytes], outfile: str):
    with wave.open(outfile, "wb") as outwave:
        for wav in wav_contents:
            with wave.open(io.BytesIO(wav), "rb") as inwave:
                if outwave.getnframes() == 0:
                    outwave.setparams(inwave.getparams())
                outwave.writeframes(inwave.readframes(inwave.getnframes()))


async def fetch_wav_async(session: aiohttp.ClientSession, url: str, semaphore: Semaphore) -> Optional[bytes]:
    async with semaphore:
        try:
            print("Getting ", url)
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    print(f"Failed to download {url}: {response.status}")
                    return None
                return await response.read()
        except Exception as e:
            print(e)
            return None


async def fetch_separate_wavs_async(session: aiohttp.ClientSession, row, semaphore: Semaphore) -> List[bytes]:
    infiles = [row["wav" + str(i) + "_Url"] for i in range(1, 7)]
    tasks = [fetch_wav_async(session, url, semaphore) for url in infiles]
    files = await asyncio.gather(*tasks)
    filtered_files = list(filter(None, files))
    assert len(filtered_files) == 6, f"Expected 6 WAV files, got {len(filtered_files)}"
    return filtered_files


async def fetch_zip_wavs_async(session: aiohttp.ClientSession, row, semaphore: Semaphore) -> List[bytes]:
    infile = row["Zip File_Url"]
    async with semaphore:
        async with session.get(infile, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"Failed to download {infile}: {response.status}")
            zip_content = await response.read()

    # Rest of zip processing remains synchronous as it's IO-bound with local files
    zip_buffer = io.BytesIO(zip_content)
    with zipfile.ZipFile(zip_buffer) as zip_ref:
        files = list(filter(lambda x: x.endswith(".wav"), zip_ref.namelist()))
        files = sorted(files, key=lambda x: int(re.findall(r"\d+", x)[0]))  # make sure they're in order
        print(f"Found WAV files in zip: {files}")

        wav_contents = []
        for wav_file in files:
            content = zip_ref.read(wav_file)
            wav_buffer = io.BytesIO(content)
            try:
                with wave.open(wav_buffer, "rb"):
                    wav_contents.append(content)
            except:
                pass
        assert len(wav_contents) == 6, f"Expected 6 WAV files, got {len(wav_contents)}"
        return wav_contents


async def fetch_row_wavs_async(session: aiohttp.ClientSession, row, semaphore: Semaphore) -> List[bytes]:
    try:
        if row["Zip File_Url"].strip().endswith("="):
            return await fetch_separate_wavs_async(session, row, semaphore)
        else:
            return await fetch_zip_wavs_async(session, row, semaphore)
    except Exception as e:
        print(f"Error processing row {row['ResponseId']}: {e}")
        return []


async def process_row_async(session: aiohttp.ClientSession, row, semaphore: Semaphore):
    try:
        wav_contents = await fetch_row_wavs_async(session, row, semaphore)
        if not wav_contents:
            return
        outfile = path.join("corpus", "unaligned", row["ResponseId"] + ".wav")
        # Run CPU-bound concat_wavs in a thread pool
        await asyncio.to_thread(concat_wavs, wav_contents, outfile)
    except Exception as e:
        print(f"Error processing row {row['ResponseId']}: {e}")


async def get_qualtrics_data_async():
    semaphore = Semaphore(MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        with open("data/survey.csv", mode="r") as csv_file:
            rows = list(csv.DictReader(csv_file))
            tasks = [process_row_async(session, row, semaphore) for row in rows]
            await asyncio.gather(*tasks)


def get_qualtrics_data():
    asyncio.run(get_qualtrics_data_async())


if __name__ == "__main__":
    get_qualtrics_data()
