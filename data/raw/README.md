Put the raw 5-minute CSV files in this directory.

Expected format:

time,open,high,low,close

Example:

1609459200,100.0,101.0,99.5,100.5

"time" must be a Unix timestamp in seconds.

The program reads all CSV files in this directory,
merges them, removes duplicate timestamps,
and resamples them into daily OHLC data.
