from google.cloud import storage
import os

# Set the environment variable for the service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.environ['GCS_ACCESS']

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a new blob and upload the file's content
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def main():
    # Define the bucket name and file details
    bucket_name = os.environ['GCS_BUCKET']
    source_file_name = 'hello_world.txt'
    destination_blob_name = 'hello_world.txt'

    # Create a text file with "Hello, World!"
    with open(source_file_name, 'w') as file:
        file.write('Hello, World!')

    # Upload the file to the bucket
    upload_blob(bucket_name, source_file_name, destination_blob_name)

if __name__ == '__main__':
    main()