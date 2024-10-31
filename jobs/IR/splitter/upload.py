import os

os.system(f'gcloud storage cp ./data/done/train.pth gs://ml-model-bucket-123456/train.pth')
os.system(f'gcloud storage cp ./data/done/val.pth gs://ml-model-bucket-123456/val.pth')
