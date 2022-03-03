# Example script to perform a HF fine-tuning logged to ClearML

## Pre-requisites:

- An account on ClearML
- The `clearml` Python client installed locally
- AWS S3 creds

## How to replicate:

1. Run `clearml-init` locally to create a local clearml config (at `~/clearml.conf`)
2. Add your AWS S3 credentials to that clearml config (so it can upload to S3)
3. Make sure you have your Task related stuff filled out in the Python script (and you will need to specify the S3 bucket/path that you want to have your output go to)
4. Run the script locally via just `python train.py ...`
5. That will log the code, commit ID, requirements, etc. to ClearML (that's the end if unless you need to run it as a job on the AQuA server)
6. If you want to run your job on the AQuA server, then
    1. Just abort the local job in the ClearML dashboard
    2. Clone the task
    3. Then enqueue the task on one of the queues
