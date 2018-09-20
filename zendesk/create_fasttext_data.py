from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import argparse
import boto3
import sys
import pandas as pd
import numpy as np
import re
from io import StringIO
import logging

logger = logging.getLogger(__name__)

DB = "s3"
CHUNK_SIZE = 10**5
COLUMNS = ['ticket_category', 'description']

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub(r"'", " ' ", string)
  string = re.sub(r"\.", " . ", string)
  string = re.sub(r"-", " - ", string)
  string = re.sub(r":", " : ", string)
  string = re.sub(r"\"", " \" ", string)
  string = re.sub(r"@", " @ ", string)
  string = re.sub(r"#", " # ", string)
  string = re.sub(r"~", " ~ ", string)
  string = re.sub(r"`", " ` ", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", " ? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()


def process_file(bucket, path, text_file_path):
  if path == None or text_file_path == None:
    return
  samples = []
  with open(text_file_path, 'w') as f:
    for chunk in pd.read_csv('{}://{}/{}'.format(DB, bucket, path), 
      encoding='utf-8',
      skipinitialspace=True,
      chunksize=CHUNK_SIZE,
      iterator=True):    
      for _, row in chunk.iterrows():
        f.write('__label__%s %s\n' % (row.ticket_category, clean_str(row.description)))

def main(params):
  process_file(params.bucket, params.path, params.text_file_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--bucket', type=str,
                      default='nlu-platform',
                      help='S3 bucket')
  parser.add_argument('--path', type=str,
                      default='zendesk/cm_zendesk_es.train.csv',
                      help='File path')
  parser.add_argument('--text_file_path', type=str,
                      default='cm_zendesk_es.train.txt',
                      help='Name of text file')
  params, _ = parser.parse_known_args()

  main(params)
