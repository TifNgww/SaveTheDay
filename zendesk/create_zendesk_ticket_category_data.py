from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import argparse
import boto3
import sys
import pandas as pd
import numpy as np
from io import StringIO
import logging

logger = logging.getLogger(__name__)

DB = "s3"
CHUNK_SIZE = 10**5
COLUMNS = ['ticket_category', 'description']


def process_file(bucket, path, train_portion, train_file_path, test_file_path):
  if path == None or train_file_path == None or test_file_path == None:
    return
  samples = []
  for chunk in pd.read_csv('{}://{}/{}'.format(DB, bucket, path), 
    encoding='utf-8',
    skipinitialspace=True,
    chunksize=CHUNK_SIZE,
    iterator=True):    
    for _, row in chunk.iterrows():
      ticket_cat = str(row.ticket_category).lower()
      desc = str(row.description).lower().strip().replace('\t', ' ')
      samples.append([ticket_cat, desc])

  n = int(len(samples) * train_portion)

  train_samples = samples[:n]
  test_samples = samples[n:]

  def to_s3(path, samples):
    logger.info('Output training samples to: {}'.format(path))
    out_df = pd.DataFrame(samples, columns=COLUMNS)
    csv_buffer = StringIO()
    out_df.to_csv(csv_buffer, encoding='utf-8')
    content = csv_buffer.getvalue()
    client = boto3.client('s3')
    client.put_object(Bucket=bucket, Key=path, Body=content)

  to_s3(train_file_path, train_samples)
  to_s3(test_file_path, test_samples)



def main(params):
  process_file(params.bucket, params.path, params.train_portion, params.train_file_path, params.test_file_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--bucket', type=str,
                      default='nlu-platform',
                      help='S3 bucket')
  parser.add_argument('--path', type=str,
                      default='zendesk/cm_zendesk_es.csv',
                      help='File path')
  parser.add_argument('--train_portion', type=float,
                      default=0.9,
                      help='Number of training samples')
  parser.add_argument('--train_file_path', type=str,
                      default='zendesk/cm_zendesk_es.train',
                      help='Name of training file')
  parser.add_argument('--test_file_path', type=str,
                      default='zendesk/cm_zendesk_es.test',
                      help='Name of test file')
  params, _ = parser.parse_known_args()

  main(params)
