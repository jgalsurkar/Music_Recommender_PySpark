# Imports
import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import *

#for SparkConf() check out http://spark.apache.org/docs/latest/configuration.html
conf = (SparkConf()
         .setMaster("local")
         .setAppName("Music")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)

def parse_artist_id(line):
	"""
		Parse a line into the good and bad ID for the artist
	"""
    try:
        idx, name = line.split('\t')
        return tuple([int(idx), name.strip()])
    except:
        pass

def parse_alias_id(line):
	"""
		Parse a line into artist id and name
	"""
    try:
        id1, id2 = line.split('\t')
        return tuple([int(id1), int(id2)])
    except:
        pass

def parse_user_artist_data(line):
	"""
		Create a Rating object given a line
	"""
    user_ID, artist_ID, count = [int(item) for item in line.split()]
    final_artist_ID = b_artist_alias.value.get(artist_ID, artist_ID)
    return Rating(user_ID, final_artist_ID, count)

if __name__ == "__main__":	
	# For AWS
	user_artist_loc  = "s3://folder/user_artist_data.txt"
	artist_data_loc  = "s3://folder/artist_data.txt"
	artist_alias_loc = "s3://folder/artist_alias.txt"

	# Read in Data
	raw_user_artist_data = sc.textFile(user_artist_loc)
	raw_artist_data     = sc.textFile(artist_data_loc)
	raw_artist_alias    = sc.textFile(artist_alias_loc)

	# Parse artist data
	artist_by_ID = raw_artist_data.map(lambda x: parse_artist_id(x)).filter(lambda x: x != None)
	artist_alias = raw_artist_alias.map(lambda x: parse_alias_id(x)).filter(lambda x: x != None).collectAsMap()
	b_artist_alias = sc.broadcast(artist_alias) # Holds a copy in memory

	# Parse User Artist Data
	train_data = raw_user_artist_data.map(lambda line: parse_user_artist_data(line)).cache()

	# Alternating Least Squares Matrix Factorization parameters
	rank = 50
	iterations = 10
	lambd = 1.0
	a = 40.0

	# Define and train the model
	model = ALS.trainImplicit(train_data, rank, iterations, lambd, alpha = a)

	# Example of making 10 recommendations
	recommendations = model.recommendProducts(2093760, 10)
	recommendedProductIDs = [recommendations[i][1] for i in range(len(recommendations))]

	recommendedArtists = artist_by_ID.filter(lambda x: x[0] in recommendedProductIDs).values().collect()
	
	for i in range(len(recommendedArtists)):
		print '{0} {1}'.format(i+1, recommendedArtists[i])

	sc.stop()