# solafs
Audio Time Scale Modification via Synchronous OverLap-Add with Fixed Synthesis hop

`usage: solafs.py [-h] [--scale SCALE] [--win WIN] [--hop HOP]
                 [--max_shift MAX_SHIFT] [--shuffle SHUFFLE]
                 [--max_duration MAX_DURATION]
                 input output

Modify WAV files with solafs.

positional arguments:
  input                 input WAV file
  output                output WAV file

optional arguments:
  -h, --help            show this help message and exit
  --scale SCALE         Factor scaling output duration.
  --win WIN             Window time in seconds.
  --hop HOP             Window hop advance in seconds.
  --max_shift MAX_SHIFT
                        Maximum time shift to synchronize.
  --shuffle SHUFFLE     SD of time over which to shuffle frames.
  --max_duration MAX_DURATION
                        Truncate input at this duration.
`
