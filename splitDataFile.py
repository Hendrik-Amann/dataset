import os
import argparse

import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as spark_types
  
def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root", type=str, help="")
  parser.add_argument("--partitions", type=int, default=500, help="")
  parser.add_argument("--memory", type=str, default="1g", help="")
  parser.add_argument("--trainSize", type=int, default=4000, help="")
  parser.add_argument("--valSize", type=int, default=500, help="")

  args, unknown = parser.parse_known_args()
  return args, unknown

def main():
  args, unknown = read_args()
  
  conf = pyspark.SparkConf()
  conf.set('spark.driver.memory', args.memory)
  sc = pyspark.SparkContext(conf=conf)
  spark = pyspark.sql.SparkSession(sc)
  
  data_path = os.path.join(args.data_root, 'selectedSamples.txt')
  df = spark.read.json(data_path).repartition(500, "article_id").orderBy(F.rand())

  train_df = df.limit(args.trainSize)
  valid_test_df = df.filter(~df["article_id"].isin(list(train_df.select(train_df.article_id).toPandas()['article_id'])))
  valid_df = valid_test_df.limit(args.valSize)
  test_df = valid_test_df.filter(~valid_test_df["article_id"].isin(list(valid_df.select(train_df.article_id).toPandas()['article_id'])))

  train_df.write.json(path=os.path.join(args.data_root, "train"), mode="overwrite")
  test_df.write.json(path=os.path.join(args.data_root, "test"), mode="overwrite")
  valid_df.write.json(path=os.path.join(args.data_root, "val"), mode="overwrite")

  os.system('cat ' + args.data_root + '/train/part-* >' + args.data_root + '/train.txt')
  os.system('cat ' + args.data_root + '/val/part-* >' + args.data_root + '/val.txt')
  os.system('cat ' + args.data_root + '/test/part-* >' + args.data_root + '/test.txt')

  os.system('rm -r ' + args.data_root + '/train')
  os.system('rm -r ' + args.data_root + '/val')
  os.system('rm -r ' + args.data_root + '/test')

if __name__ == "__main__":
  main()
