import configparser
import pyspark.sql.functions as F
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Creates a Spark session with required configurations

    Returns:
        SparkSession: Spark session instance
    """
    # Create a Spark session and configure required packages
    return SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()


def process_song_data(spark, input_data, output_data):
    """
    This function reads song data from a json file, processes the data and writes it to parquet files in the output directory.
    It creates two tables - 'songs' and 'artists' partitioned by year and artist_id respectively.
    The song data is read from the input directory.

    Parameters:
    spark (SparkSession): The SparkSession object used to access Spark functions
    input_data (str): The directory path to the input data file
    output_data (str): The directory path to the output directory

    Returns:
    None
    """
    try:
        # Get the file path to the song data file
        song_data = input_data + "song_data/*/*/*/"

        # Read the song data file into a Spark DataFrame
        df = spark.read.json(song_data)

        # Extract the relevant columns to create the songs table
        songs_table = df.select(["song_id", "title", "artist_id", "year", "duration"]).dropDuplicates()

        # Write the songs table to parquet files, partitioned by year and artist
        songs_table.write.partitionBy(['year', 'artist_id']).parquet(output_data + 'songs/songs.parquet')

        # Extract the relevant columns to create the artists table
        artists_table = df.select(
            ["artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude"]).dropDuplicates()

        # Write the artists table to parquet files, partitioned by artist
        artists_table.write.partitionBy(['artist_id']).parquet(output_data + 'artists/artists.parquet')
    except Exception as e:
        # Print an error message in case of exception
        print("Error: ", e)


def process_log_data(spark, input_data, output_data):
    """
    This function processes log data, reads JSON files from the given input directory,
    filters the data to keep only "NextSong" events, and writes the processed data to the given output directory.

    Parameters:
    - spark (SparkSession): Spark session object.
    - input_data (str): Input directory for log data.
    - output_data (str): Output directory to write the processed data.

    Returns:
    None. Writes the processed data to output directory.
    """
    try:
        # Input log data file path
        log_data = input_data + 'log_data/*/*/*.json'

        # Read log data from JSON files
        df = spark.read.json(log_data)

        # Filter to keep only "NextSong" events
        df = df.filter(df['page'] == 'NextSong')

        # Create a table of unique users by selecting relevant columns
        artists_table = df.select(['userId', 'firstName', 'lastName', 'gender', 'level']).distinct()

        # Write the user table to output directory
        artists_table.write.parquet(f'{output_data}users/users.parquet', partitionBy=['userId'])

        # Cast the timestamp to correct format
        df = df.withColumn('timestamp', (df.ts / 1000).cast('timestamp'))

        # Create time table by selecting relevant columns and extracting various time related fields
        time_table = df.select(
            F.col("timestamp").alias('start_time'),
            F.hour('timestamp').alias('hour'),
            F.dayofmonth('timestamp').alias('day'),
            F.weekofyear('timestamp').alias('week'),
            F.month('timestamp').alias('month'),
            F.year('timestamp').alias('year'),
            F.date_format('timestamp', 'E').alias('weekday')
        )
        # Write the time table to output directory
        time_table.write.parquet(f'{output_data}time/time.parquet', partitionBy=['start_time'])

        # Read song data from JSON files
        song_df = spark.read.json(input_data + 'song_data/*/*/*/*.json')

        # Join the log data and song data on relevant columns to create a songplays table
        song_log_joined_table = df.join(
            song_df,
            (df.song == song_df.title) & (df.artist == song_df.artist_name) & (df.length == song_df.duration),
            'inner'
        )

        # Select relevant columns from the joined table and rename columns for better readability
        songplays_table = song_log_joined_table \
            .distinct() \
            .select(
                'userId', 'timestamp', 'song_id', 'artist_id', 'level', 'sessionId', 'location', 'userAgent'
            ) \
            .withColumn('songplay_id', F.row_number().over(Window.partitionBy('timestamp').orderBy('timestamp'))) \
            .withColumnRenamed('userId', 'user_id') \
            .withColumnRenamed('timestamp', 'start_time') \
            .withColumnRenamed('sessionId', 'session_id') \
            .withColumnRenamed('userAgent', 'user_agent')
        songplays_table.write.parquet(f'{output_data}songplays/songplays.parquet', partitionBy=['start_time', 'user_id'])
    except Exception as e:
        # Print an error message in case of exception
        print("Error: ", e)

def main():
    """
    Main function to process the song and log data and save the processed data to output directory

    Returns:
        None
    """
    # Create a Spark session
    spark = create_spark_session()

    # Input data location
    input_data = "s3a://udacity-dend/"

    # For local testing: input_data = "data/input/"

    output_data = " "

    # Output data location
    # For local testing: output_data = "data/output/"

    # Process the song data
    process_song_data(spark, input_data, output_data)

    # Process the log data
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
