import pstats
from sys import argv

# Get the profile file name from the command line arguments
profile_file = argv[1]

# Read the profile data
p = pstats.Stats(profile_file)

# Sort the statistics by the cumulative time spent in the function
p.sort_stats('cumulative')

# Print the statistics of the top 10 functions
p.print_stats(100)
