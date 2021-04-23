"""
CLI tool to downloads OSM data asynchronously
"""

import os
import time
import json
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from osm_tools import _bbox_from_point, DEFAULT_OVERPASS_URL, DEFAULT_QUERY_TEMPLATE


async def fetch(session, index, query, contact, retry_limit=5):
    url = DEFAULT_OVERPASS_URL
    headers = {"user-agent": "capstone-geodome/0.2", "from": contact}
    async with session.get(url, params={"data": query}, headers=headers) as response:
        retries = 0
        data = np.nan

        while retries < retry_limit:
            try:
                response_content = await response.read()
                data = json.loads(response_content)
                if response.status == 200:
                    break
            except Exception:
                retries += 1
                # TODO: does the response status carry the same info?
                if "The server is probably too busy to handle your request" in str(
                    response_content
                ):
                    await asyncio.sleep(2)
                    pass
                pass
            pass
        return index, data


async def osm_async_download(bbox_list, template, contact):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch(session, index, template.format(bbox), contact)
            for (index, bbox) in bbox_list
        ]
        responses = [
            await t
            for t in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="download",
                leave=False,
            )
        ]
    return responses


def get_tags_col(df, distance, lat_col, lon_col, contact):
    bbox_list = [
        _bbox_from_point((lat, lon), dist=distance)
        for (lat, lon) in zip(df[lat_col], df[lon_col])
    ]
    bbox_list = [(i, bbox) for (i, bbox) in enumerate(bbox_list)]
    indexed_response = asyncio.run(
        osm_async_download(bbox_list, DEFAULT_QUERY_TEMPLATE, contact)
    )
    indexed_response.sort(key=lambda tup: tup[0])
    assert len(indexed_response) == len(df)

    return [r for (_, r) in indexed_response]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="path to csv input file", type=str)
    parser.add_argument(
        "-o",
        "--outfile",
        help="path to csv output file",
        default="osm_out.csv",
        type=str,
    )
    parser.add_argument(
        "-ce",
        "--contact_email",
        help="contact email to include in the request header",
        default="TBD",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--distance",
        help="side length for area of interest",
        default=550,
        type=int,
    )
    parser.add_argument(
        "-lat",
        "--lat_col",
        help="name of the column that contains the latitude",
        default="lat",
        type=str,
    )
    parser.add_argument(
        "-lon",
        "--lon_col",
        help="name of the column that contains the longitude",
        default="lon",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--splits",
        help="number of splits to make to the input file",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-t",
        "--pause_time",
        help="number of seconds to wait between request batches",
        default=60,
        type=int,
    )
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    df_parts = np.array_split(input_df, args.splits)
    for i, part in enumerate(tqdm(df_parts, desc="splits")):
        tmpfname = "{}_p{}.csv".format(args.outfile.split(".")[0], i)
        if os.path.exists(tmpfname):
            continue
        part["tags"] = get_tags_col(
            part, args.distance, args.lat_col, args.lon_col, args.contact_email
        )
        part.to_csv(tmpfname, index=False)
        time.sleep(args.pause_time)

    outfile = pd.concat(
        [
            pd.read_csv("{}_p{}.csv".format(args.outfile.split(".")[0], i))
            for i in range(args.splits)
        ],
        ignore_index=True,
    )
    # if some of the requests are missed, retry one more time with a new connection
    if outfile["tags"].isna().sum() > 0:
        print(
            "second pass for requests that the surver was too busy to fulfill the first time"
        )
        na_index = outfile[outfile["tags"].isna()].index
        outfile.loc[na_index, ["tags"]] = get_tags_col(
            outfile[outfile["tags"].isna()],
            args.distance,
            args.lat_col,
            args.lon_col,
            args.contact_email,
        )
        pass
    outfile.to_csv(args.outfile, index=False)
