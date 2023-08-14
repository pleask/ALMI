import pstats 

profiler = pstats.Stats('profile.out')
profiler.strip_dirs()
profiler.sort_stats('tottime')

# Get the top functions
top_functions = profiler.print_stats()