"""S3 client utilities for data storage and retrieval."""

import json
import boto3
import pandas as pd
from io import BytesIO
from typing import Optional, Dict, Any


class S3Client:
    """S3 client wrapper for investment system operations."""

    def __init__(self, bucket: str, region: str = 'us-east-1'):
        self.bucket = bucket
        self.s3 = boto3.client('s3', region_name=region)

    def read_parquet(self, key: str) -> pd.DataFrame:
        """Read a parquet file from S3."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            buffer = BytesIO(response['Body'].read())
            return pd.read_parquet(buffer)
        except self.s3.exceptions.NoSuchKey:
            return pd.DataFrame()
        except Exception as e:
            print(f"Error reading parquet from {key}: {e}")
            return pd.DataFrame()

    def write_parquet(self, df: pd.DataFrame, key: str) -> bool:
        """Write a DataFrame as parquet to S3."""
        try:
            buffer = BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())
            return True
        except Exception as e:
            print(f"Error writing parquet to {key}: {e}")
            return False

    def read_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Read a JSON file from S3."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            print(f"Error reading JSON from {key}: {e}")
            return None

    def write_json(self, data: Dict[str, Any], key: str) -> bool:
        """Write a dict as JSON to S3."""
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(data, indent=2, default=str),
                ContentType='application/json'
            )
            return True
        except Exception as e:
            print(f"Error writing JSON to {key}: {e}")
            return False

    def append_jsonl(self, data: Dict[str, Any], key: str) -> bool:
        """Append a JSON line to a JSONL file in S3."""
        try:
            # Read existing content
            try:
                response = self.s3.get_object(Bucket=self.bucket, Key=key)
                existing = response['Body'].read().decode('utf-8')
            except self.s3.exceptions.NoSuchKey:
                existing = ''

            # Append new line
            new_line = json.dumps(data, default=str) + '\n'
            updated = existing + new_line

            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=updated.encode('utf-8'),
                ContentType='application/x-ndjson'
            )
            return True
        except Exception as e:
            print(f"Error appending JSONL to {key}: {e}")
            return False

    def read_csv(self, key: str) -> pd.DataFrame:
        """Read a CSV file from S3."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pd.read_csv(BytesIO(response['Body'].read()))
        except self.s3.exceptions.NoSuchKey:
            return pd.DataFrame()
        except Exception as e:
            print(f"Error reading CSV from {key}: {e}")
            return pd.DataFrame()

    def file_exists(self, key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False

    def list_keys(self, prefix: str) -> list:
        """List all keys with a given prefix."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            print(f"Error listing keys with prefix {prefix}: {e}")
            return []
