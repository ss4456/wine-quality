Name: Sukriti Singh
NJIT-ID: ss4456
CS643 Programming Assignment 2

This README.txt file contains instructions to train machine learning model for 
estimating wine quality (using the data provided).

Code is available on GitHub (https://github.com/ss4456/wine-quality)
Docker Hub image is available here (public link): https://hub.docker.com/r/ss4456/wine-quality

SETTING UP THE CLOUD ENVIRONMENT
======================================================
Step 1: Update the flintrock config file
- Update the PEM file
- Set number of slaves to 4.
- Set HDFS and spark install to True
- Set AMI based on EC2 instance (previously created)

Step 2: Run the following commands to set up the cluster
> flintrock launch ss4456_cluster
> flintrock run-command ss4456_cluster 'sudo yum install -y gcc'
> flintrock run-command ss4456_cluster 'pip3 install numpy'
> flintrock run-command ss4456_cluster 'pip3 install findspark'

MODEL TRAINING
======================================================
After experimentation with different machine learning model - Linear Regression, 
Logisitic Regression, Random Forests, the best performance on validation dataset
was achieved using Random Forests (F1 0.57, compared to 0.48 and 0.52 with Linear 
and Logistic models). I experimented with both standard normalization and min max 
scaling to preprocess the data; they didn't bring significant improvements, however,
the standard normalization was marginally better, and hence is included in the final
training code (on GitHub).

Step 1: Secure copy (or git clone) the code and data from local system to the master node

Step 2: Log onto the cluster
> flintrock login ss4456_cluster

Step 3: Copy the data on the master node to the HDFS
> hadoop fs -mkdir /data
> hadoop fs -put ./TrainingDataset.csv /data
> hadoop fs -put ./ValidationDataset.csv /data

Step 4: Launch training
> ./spark/bin/spark-submit train_rf.py 

Step 5: Download the trained model (preprocessing and machine learning models)
> hadoop fs -get /models models

MODEL TESTING ON THE CLUSTER (non-Docker mode)
======================================================
Testing the trained model on a dataset (ValidationDataset.csv)
> ./spark/bin/spark-submit \
    --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties" \
    --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties" \
    --files "./log4j.properties" \
    predict.py /data/ValidationDataset.csv
Above settings help hide the warning logs, so the standard output is easily readable. 
The log4h.properties file required to run the above command is available on the GitHuB.
Expected output: 
"F measure: 0.5743" (rounded for readability)

DOCKERIZATION
======================================================
Step 1: Log on to the master node 
> flintrock login ss4456_cluster

Step 2: Run following commands to install docker
> sudo yum install docker -y
> sudo service docker start
> sudo usermod -a -G docker ec2-user

Step 3: Followed steps to build the docker image and push it to the public hub.

MODEL TESTING (Docker mode)
======================================================
Step 1: Copy the dataset CSV file to the EC2 instance
> scp <local-csv-file> <EC2-path>

Step 2: Log on the EC2 instance. Here, I used the master node for testing,
> flintrock login ss4456_cluster

Step 3: Pull the image from the docker hub (unless it's already stored locally)
> docker login (enter login credentials)
> docker pull ss4456:wine-quality/version1

Step 4: Process the CSV file using the model on the docker container
> docker run -v /home/ec2-user:/localdata  ss4456:wine-quality/version1 python3 predict.py /localdata/<local-csv-file>

As a test example, run the following command
> docker run -v /home/ec2-user:/localdata  ss4456:wine-quality/version1 python3 predict.py /localdata/ValidationDataset.csv
Expected output: 
"F measure: 0.5743" (rounded for readability)
