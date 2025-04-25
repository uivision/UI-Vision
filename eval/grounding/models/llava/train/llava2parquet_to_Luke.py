import json
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def process_item(item):
    # Read the image and convert it to byte format
    with open(item["image"], "rb") as img_file:
        img_bytes = img_file.read()

    record = {
        "image": img_bytes,
        "conversations": json.dumps(item["conversations"])  # Serialize as JSON string
    }
    return record


# Read the JSON file
with open('merged_half.json', 'r') as file:
    data = json.load(file)

local_path = 'merged_first_half.parquet'

# Get the number of CPU cores in the system
cpu_count = os.cpu_count()

# Process data in batches
batch_size = 100000  # Can be adjusted based on actual needs
num_batches = (len(data) + batch_size - 1) // batch_size

# Local file path
# local_path = 'final_data_4ch.parquet'

# Initialize ParquetWriter
with open(local_path, 'wb') as local_file:
    writer = None

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(data))
        batch_data = data[start_index:end_index]

        # Use ThreadPoolExecutor for parallel processing
        records = []
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            future_to_record = {executor.submit(process_item, item): item for item in batch_data}
            for future in tqdm(as_completed(future_to_record), total=len(future_to_record),
                               desc=f"Processing Batch {batch_index + 1}/{num_batches}"):
                try:
                    record = future.result()
                    records.append(record)
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

        # Create a PyArrow table
        table = pa.Table.from_pandas(pd.DataFrame(records))

        # If it's the first batch, set the writer and schema
        if writer is None:
            writer = pq.ParquetWriter(local_file, table.schema, version='2.6', use_dictionary=True, compression='snappy')

        # Write to the Parquet file in chunks
        for i in tqdm(range(0, len(table), 4), desc=f"Writing Batch {batch_index + 1}/{num_batches} to Parquet"):
            writer.write_table(table.slice(i, 4))

    writer.close()

print("Completed: Batches saved as Parquet files to local directory")
