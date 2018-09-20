"""
Process csv file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import boto3
import sys
import langid
import re

import pandas as pd

from io import StringIO
from collections import defaultdict


DB = "s3"
COLUMNS = ["type", "ticket_category", "ticket_type", "description"]
LABEL_BLACK = set(["nan", "other"])
CHUNK_SIZE = 10**5


def text_norm(text):
    return ' '.join(filter(None, 
            [segment.strip() for segment in re.split("[-\\n\\r]+", text)]))


def process_file(bucket, path, lang_filter=None, output=None):
    if lang_filter is not None:
        assert output is not None, "Output file should be specified"

    # dataframe
    type_stat, ticket_cat_stat, ticket_type_stat, lang_stat = \
        defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    
    ln = 0 # line number
    out_df = []
    for chunk in pd.read_csv('{}://{}/{}'.format(DB, bucket, path), 
                             encoding='utf-8',
                             skipinitialspace=True,
                             usecols=COLUMNS,
                             chunksize=CHUNK_SIZE,
                             iterator=True):
        for _, row in chunk.iterrows():
            ln += 1
            if ln % 10000 == 0:
                print('Processed {} lines'.format(ln))

            row_type = str(row.type).lower()
            ticket_cat = str(row.ticket_category).lower()
            if ticket_cat in LABEL_BLACK:
                continue
            
            ticket_type = str(row.ticket_type).lower()
            desc = str(row.description).lower().strip()
            desc = text_norm(desc)
            if not desc:
                continue

            lid, conf = langid.classify(desc)
            if conf > 0:
                continue

            type_stat[row_type] += 1
            ticket_cat_stat[ticket_cat] += 1
            ticket_type_stat[ticket_type] += 1
            lang_stat[lid] += 1

            if output is not None and lid == lang_filter:
                out_df.append([row_type, ticket_cat, ticket_type, desc])

    print('Type: {}'.format(dict(type_stat)))
    print('Ticket Category: {}'.format(dict(ticket_cat_stat)))
    print('Ticket Type: {}'.format(dict(ticket_type_stat)))
    print('Languages: {}'.format(dict(lang_stat)))

    if out_df:
        print('Output to: {} ({})'.format(output, len(out_df)))
        out_df = pd.DataFrame(out_df, columns=COLUMNS)
        csv_buffer = StringIO()
        out_df.to_csv(csv_buffer, encoding='utf-8')
        # s3_resource = boto3.resource(DB)
        # s3_resource.Object(bucket, output).put(Body=csv_buffer.getvalue())
        content = csv_buffer.getvalue()

        def to_s3(bucket, output, content):
            client = boto3.client('s3')
            client.put_object(Bucket=bucket, Key=output, Body=content)

        to_s3(bucket, output, content)


def main(params):
    process_file(params.bucket, params.path, params.lang_filter, params.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str,
                        default='nlu-platform',
                        help='S3 bucket')
    parser.add_argument('--path', type=str,
                        default=None,
                        help='File path')
    parser.add_argument('--lang_filter', type=str,
                        default=None,
                        help='Language to be filtered out')
    parser.add_argument('--output', type=str,
                        default=None,
                        help='Output file path for the filtered language')

    params, _ = parser.parse_known_args()
    main(params)
