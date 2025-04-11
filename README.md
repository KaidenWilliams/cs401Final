# cs401Final


Version control for the code part of the cs401 final project + shared document for info

YOLO

Architecture Diagram:

![image](https://github.com/user-attachments/assets/9046be0f-b5c3-44ae-a3c8-782bcc3d6a9c)
  <br>  

Pipeline Text Walkthrough: - TODO

1. New Data Uploaded to raw data s3 bucket
2. Eventbridge? sees the data upload, calls Glue Training Job
4. Glue training job runs, completes, uploads transformed data to different s3 bucket
5. Eventbrige? sees transformed data uploaded to s3, calls SageMaker Pipeline
6. Sagemaker pipeline does the following steps
7. Does additional EDA, uploads to feature store
8. Gets data from feature store, splits into train / test, uploads that to s3
9. Train / test used to train model
10. Model uploaded to model registry
11. Model deployed as endpoint, autoscaling is enabled
12. Somehow API Gateway automatically gets made / configured
13. Model monitor scheduled job is deployed?
Note: we also need to configure all out code with Github, get things version controlled / integrated with Github


TODO:

1. Data Preparation (getting it in s3) - Kaiden/Nathan
2. EDA / Data Cleaning - Kaiden/Nathan
3. Making Model - Spencer/Jacob
4. Making Endpoint - TBD
5. Connecting whole thing to Github - TBD
6. Making whole thing into Automatic Pipeline (EventBridge, SageMaker Pipelines / Jobs, might be kinda hard) - TBD
<br>

Helpful Links:

- [2021 1st Place Article Explanation](https://github.com/namakemono/kaggle-birdclef-2021/tree/master)
- [2024 1st Place Article Explanation](https://www.kaggle.com/competitions/birdclef-2024/discussion/512197)
- [2024 1st Place YouTube Explanation](https://www.youtube.com/watch?v=6o6wGm25lA0)
- [EDA and Training Code for Kaggle 2024 Solution](https://github.com/skj092/kaggle-BirdCLEF-2024/blob/main/notebooks/training.ipynb)
- [2024 1st Place Inference Code](https://www.kaggle.com/code/chemrovkirill/birdclef-2024-1st-place-inference)
- [Kaggle Discussion about Human Voices](https://www.kaggle.com/competitions/birdclef-2025/discussion/568886)

