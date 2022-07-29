
import findspark
findspark.init()

import argparse

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel

def remove_quotes(sdf):
    for col in sdf.columns:
        sdf = sdf.withColumnRenamed(col, col.replace("\"", ""))
    return sdf

def main(args):
    
    sdf = spark.read.csv(args.testfile, header=True, inferSchema=True, sep=';')
    sdf = remove_quotes(sdf)

    preproc = PipelineModel.load('/models/preproc.model')
    sdf = preproc.transform(sdf)

    rf_model = RandomForestClassificationModel.load(args.model)
    predictions = rf_model.transform(sdf)

    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
    f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    print('F measure: {}'.format(f1))

# Spark Context
spark = SparkSession.builder.appName('ml-wine-quality').getOrCreate()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('testfile', type=str, help='Path to the CSV file with samples to predict')
    parser.add_argument('--model', type=str, help='Path to the machine learning model', 
                        default="/models/mllib-wine-quality.model")
    args = parser.parse_args()

    main(args)



