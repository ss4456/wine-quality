
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

def remove_quotes(sdf):
    for col in sdf.columns:
        sdf = sdf.withColumnRenamed(col, col.replace("\"", ""))
    return sdf

# Spark Context
spark = SparkSession.builder.appName('ml-wine-quality').getOrCreate()

# Spark dataframes for training and validation data
sdf_train = spark.read.csv(r'/data/TrainingDataset.csv', header=True, inferSchema=True, sep=';')
sdf_valid = spark.read.csv(r'/data/ValidationDataset.csv', header=True, inferSchema=True, sep=';')

sdf_train = remove_quotes(sdf_train)
sdf_valid = remove_quotes(sdf_valid)

feature_names = [c for c in sdf_train.columns if c != 'quality']

assembler = VectorAssembler(inputCols=feature_names, outputCol="features_assembled")
normalizer = StandardScaler(inputCol='features_assembled', outputCol='features')
pipeline = Pipeline(stages=[assembler, normalizer])

pipeline_model = pipeline.fit(sdf_train)
pipeline_model.write().overwrite().save('/models/preproc.model')

sdf_train = pipeline_model.transform(sdf_train)
sdf_valid = pipeline_model.transform(sdf_valid)

rf = RandomForestClassifier(impurity="gini", maxDepth=5, numTrees=10, 
                            featureSubsetStrategy="auto", seed=42, featuresCol="features", 
                            labelCol="quality")

rf_model = rf.fit(sdf_train)
rf_model.write().overwrite().save('/models/mllib-wine-quality.model')

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")

predictions = rf_model.transform(sdf_valid)

f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

with open(r'./train_output.txt', 'w') as f:
    f.write('F measure: {}\n'.format(f1))

print('F measure: {}'.format(f1))
