import asyncio
import csv
import io
import os
import wave
import zipfile
from os import path
from typing import List, Optional

import aiohttp

headers = {"X-API-TOKEN": os.environ["qualtrics_key"]}


async def fetch_separate_wavs(row) -> List[bytes]:
    async with aiohttp.ClientSession(headers=headers) as session:

        async def fetch_wav(url) -> Optional[bytes]:
            try:
                print("Getting ", url)
                async with session.get(url) as response:
                    if response.status != 200:
                        print(f"Failed to download {url}: {response.status}")
                        return None
                    return await response.read()
            except Exception as e:
                print(e)
                return None

        infiles = [row["wav" + str(i) + "_Url"] for i in range(1, 7)]
        files = await asyncio.gather(*[fetch_wav(url) for url in infiles])
        filtered_files = list(filter(None, files))
        assert len(filtered_files) == 6, f"Expected 6 WAV files, got {len(filtered_files)}"
        return filtered_files


async def fetch_zip_wavs(row) -> List[bytes]:
    infile = row["Zip File_Url"]
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(infile) as response:
            if response.status != 200:
                print(f"Failed to download {infile}: {response.status}")
                raise Exception(f"Failed to download {infile}: {response.status}")
            zip_content = await response.read()

    zip_buffer = io.BytesIO(zip_content)
    with zipfile.ZipFile(zip_buffer) as zip_ref:
        files = list(filter(lambda x: x.endswith(".wav"), zip_ref.namelist()))
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


async def fetch_row_wavs(row) -> List[bytes]:
    try:
        if row["Zip File_Url"].strip().endswith("="):
            return await fetch_separate_wavs(row)
        else:
            return await fetch_zip_wavs(row)
    except Exception as e:
        print(f"Error processing row {row['ResponseId']}: {e}")
        return []


async def concat_wavs(wav_contents: List[bytes], outfile: str) -> None:
    params = []
    frames = bytes()

    for i, content in enumerate(wav_contents):
        try:
            print(f"Reading wave data from file {i+1}")
            with wave.open(io.BytesIO(content), "rb") as w:
                frames += w.readframes(w.getnframes())
                params.append(w.getparams())
        except Exception as e:
            print(f"Error processing file {i+1}: {e}")
            continue

    with wave.open(outfile, "wb") as output:
        output.setparams(params[0])
        output.writeframes(frames)


async def get_qualtrics_data() -> None:
    async def process_row(row):
        wav_contents = await fetch_row_wavs(row)
        if not wav_contents:
            return

        outfile = path.join("data", "responses", row["ResponseId"] + ".wav")
        await concat_wavs(wav_contents, outfile)

    with open("data/survey.csv", mode="r") as csv_file:
        rows = list(csv.DictReader(csv_file))
        await asyncio.gather(*[process_row(row) for row in rows])


if __name__ == "__main__":
    asyncio.run(get_qualtrics_data())
    # print(len(os.listdir("data/responses")))
