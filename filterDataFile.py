import os
import argparse
import re

import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as spark_types

# Dancer generation does not include literature and limitations sections; filter files where can get at least 1 section
KEYWORDS = {
    'introduction': 'i',
    'case': 'i',
    'purpose': 'i',
    'objective': 'i',
    'objectives': 'i',
    'aim': 'i',
    'summary': 'i',
    #'findings': 'l',
    'background': 'i',
    'background/aims': 'i',
    #'literature': 'l',
    #'studies': 'l',
    #'related': 'l',
    'methods': 'm',
    'method': 'm',
    'techniques': 'm',
    'methodology': 'm',
    'results': 'r',
    'result': 'r',
    'experiment': 'r',
    'experiments': 'r',
    'experimental': 'r',
    'discussion': 'c',
    #'limitations': 'd',
    'conclusion': 'c',
    'conclusions': 'c',
    'concluding': 'c'}

#HA: the function below is based on section_identify method from DANCER code
#HA: instead of assigning a section type, this method checks if at least 1 section can be found based on the keyword search
def section_match(keywords):
    def section_match_(sections):
        match = False
        for section in sections:
            section = section.lower().split()
            for wrd in section:
                try:
                    match = KEYWORDS[wrd]
                except KeyError:
                    continue
        return 1 if match else 0
    return F.udf(section_match_, spark_types.ByteType())

def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root", type=str, help="")
  parser.add_argument("--partitions", type=int, default=500, help="")
  parser.add_argument("--memory", type=str, default="1g", help="")

  args, unknown = parser.parse_known_args()
  return args, unknown


def main():
  args, unknown = read_args()

  output_dir = os.path.join(args.data_root, "filter")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
      
  conf = pyspark.SparkConf()
  conf.set('spark.driver.memory', args.memory)
  sc = pyspark.SparkContext(conf=conf)
  spark = pyspark.sql.SparkSession(sc)

  b_keywords = sc.broadcast(KEYWORDS)

  df = spark.read.json(os.path.join(args.data_root, "countedTokens.txt")).repartition(args.partitions, "article_id")

  allArticles = df.count()
  df = df.where(F.col("LEDtextT") <= 16384)
  df = df.where(F.col("PXtextT") <= 16384)
  rowsTokenFilter = df.count()

  print("filter 16384")
  df = df.withColumn("match", section_match(b_keywords)("section_names")).where(F.col("match") == True).drop("match")
  print("filter match")
  rowsSectionFilter = df.count()
  df = df.orderBy(F.col("LEDtextT"), F.col("PXtextT"), ascending=False).limit(5000).drop("LEDtextT", "PXtextT").orderBy(F.rand())
  print("filter 5000")
  df.write.json(path=output_dir, mode="overwrite")

  os.system("cat " + output_dir + "/part-* >" + args.data_root + "/selectedSamples.txt")
  os.system("rm -r " + output_dir)

  print(os.path.join(args.data_root,"filterLog.txt"))
  print(os.path.join(args.data_root))
  print(args.data_root)
  with open(os.path.join(args.data_root, "filterLog.txt"), "w+") as f:
    f.write("Total number of articles in original val and test: ")
    f.write(str(allArticles))
    f.write("\n")
    f.write("Number of articles <= 16384 tokens: ")
    f.write(str(rowsTokenFilter))
    f.write("\n")
    f.write("Number of articles with DANCER section match: ")
    f.write(str(rowsSectionFilter))
    f.write("\n")

if __name__ == "__main__":
  main()
