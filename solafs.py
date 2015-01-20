"""Audio time-scale modification and scrambling via synchronous overlap-add.

   2015-01-18 Dan Ellis dpwe@ee.columbia.edu
"""

from __future__ import print_function

import argparse
import numpy as np
import scipy.io.wavfile

def Solafs(input_waveform, frame_size, overlap_size, max_shift, 
           input_point_from_output_point_fn):
  """Perform synchronous overlap-add with fixed synthesis step.
  
  Args:
    input_waveform: np.array of <frames x channels> of source waveform.
    frame_size: Number of points in each fixed synthesis step.
    overlap_size: Number of points in overlap between successive 
      synthesis windows.
    max_shift: Largest allowable shift away from the ideal input point 
      to inspect when searching for best alignment.
    input_from_output_fn: To find the next window's worth of output 
      waveform we start from 
      ideal_input_point = input_point_from_output_point_fn(
          current_output_point, input_length).

  Returns:
    output_waveform: Synthesized waveform, same number of channels 
      as input_waveform.
  """
  print("frame_size=", frame_size, " overlap_size=", overlap_size, 
        "max_shift=", max_shift)
  # Create the crossfade window.
  crossfade_window_high_to_low = (0.5 * (1.0 + 
                                         np.cos(np.pi * 
                                                np.arange(overlap_size) / 
                                                overlap_size)))[:, np.newaxis]
  print("crossfade_window.shape=", crossfade_window_high_to_low.shape)
  # Initialize output waveform; we'll extend it dynamically.
  input_frames = input_waveform.shape[0]
  if len(input_waveform.shape) > 1:
    channels = input_waveform.shape[1]
  else:
    input_waveform = np.reshape(input_waveform, (input_frames, 1))
    channels = 1
  output_waveform = np.empty(((frame_size*64), channels))
  # Pre-fill first window
  input_base_point = input_point_from_output_point_fn(0, input_frames)
  output_waveform[:frame_size] = input_waveform[input_base_point : 
                                                input_base_point + frame_size]
  # Loop through adding new windows from input.
  current_output_point = frame_size
  while True:
    ideal_input_point = input_point_from_output_point_fn(current_output_point, 
                                                         input_frames)
    if ideal_input_point is None:
      break
    ideal_input_point = max(max_shift, ideal_input_point)
    print("current_output_point=", current_output_point, 
          " ideal_input_point=", ideal_input_point)
    aligned_input_point = _FindBestAlignment(
        output_waveform[current_output_point - overlap_size : 
                        current_output_point], 
        input_waveform[ideal_input_point - max_shift : 
                       ideal_input_point + max_shift + overlap_size])
    if aligned_input_point is None:
      break
    else:
      aligned_input_point += ideal_input_point - max_shift
      #print("aligned_input_point=", aligned_input_point)
    if aligned_input_point + frame_size > input_frames:
      break
    output_overlap_range = np.arange(
        current_output_point - overlap_size, current_output_point)
    if current_output_point + frame_size > output_waveform.shape[0]:
      # Double the length of output_waveform if it filled up.
      output_waveform = np.vstack([
          output_waveform, np.empty((output_waveform.shape[0], channels))])
    # Crossfade into region of overlap.
    output_waveform[output_overlap_range] = (
        crossfade_window_high_to_low * output_waveform[output_overlap_range] +
        (1 - crossfade_window_high_to_low) * 
        input_waveform[aligned_input_point : 
                       aligned_input_point + overlap_size])
    # Copy across remainder of new frame
    output_waveform[current_output_point : 
                    current_output_point + frame_size - overlap_size] = (
        input_waveform[aligned_input_point + overlap_size : 
                       aligned_input_point + frame_size])
    current_output_point += frame_size - overlap_size
  return output_waveform[:current_output_point]


def _InputPointFromOutputPointFnTimeScaling(output_duration_ratio):
  """Return function that can be used as input_point_from_output_point_fn."""
  return (lambda output_point, input_length: 
          None if output_point > output_duration_ratio * input_length 
          else int(round(output_point / output_duration_ratio)))


def _InputPointFromOutputPointFnTimeBlur(blur_radius):
  """Return function that can be used as input_point_from_output_point_fn."""
  return (lambda output_point, input_length: 
          None if output_point >  input_length 
          else min(input_length, max(0, int(round(output_point + 
                                                  blur_radius * 
                                                  np.random.randn(1))))))


def _CosineSimilarity(vec_a, vec_b):
  """Calculate cosine similarity between two equal-sized vectors."""
  return np.sum(vec_a * vec_b) / np.sqrt(np.sum(vec_a**2)*np.sum(vec_b**2))



def _FindBestAlignment(overlap_waveform, source_waveform, 
                       alignment_fn=_CosineSimilarity):
  """Find start point of maximum correlation between overlap and source."""
  #print("FBA: ola_wv.shape=", overlap_waveform.shape, 
  #      "src_wv.shape=", source_waveform.shape)
  overlap_size, channels = overlap_waveform.shape
  num_shifts = source_waveform.shape[0] - overlap_size + 1
  alignment_scores = np.empty(num_shifts)
  for shift in np.arange(num_shifts):
    alignment_scores[shift] = alignment_fn(
        overlap_waveform, source_waveform[shift : shift + overlap_size])
  return np.argmax(alignment_scores)


def main(argv):
  """Main routine to modify a wav file using solafs."""
  parser = argparse.ArgumentParser(description="Modify WAV files with solafs.")
  parser.add_argument('input', type=str, help="input WAV file")
  parser.add_argument('output', type=str, help="output WAV file")
  parser.add_argument('--scale', type=float, 
                      help="Factor scaling output duration.")
  parser.add_argument('--win', type=float, default=0.025, 
                      help="Window time in seconds.")
  parser.add_argument('--hop', type=float, default=0.010, 
                      help="Window hop advance in seconds.")
  parser.add_argument('--max_shift', type=float, default=0.015, 
                      help="Maximum time shift to synchronize.")
  parser.add_argument('--shuffle', type=float, default=0.0, 
                      help="SD of time over which to shuffle frames.")
  parser.add_argument('--max_duration', type=float, default=0.0, 
                      help="Truncate input at this duration.")

  args = parser.parse_args()
  window_sec = args.win
  hop_sec = args.hop
  max_shift = args.max_shift
  shuffle_time = args.shuffle
  time_factor = args.scale
  max_duration = args.max_duration

  sr, data = scipy.io.wavfile.read(args.input)
  input_duration = len(data)/float(sr)
  if max_duration > 0.0 and input_duration > max_duration:
    data = data[:int(round(max_duration * sr))]
  data = data.astype(float) / 32768

  if shuffle_time > 0.0:
    time_mapping_fn = _InputPointFromOutputPointFnTimeBlur(
        int(round(shuffle_time * sr)))
  else:
    time_mapping_fn = _InputPointFromOutputPointFnTimeScaling(
        time_factor)

  data_out = Solafs(data, 
                    int(round(window_sec * sr)), 
                    int(round(hop_sec * sr)), 
                    int(round(max_shift * sr)), 
                    time_mapping_fn)

  scipy.io.wavfile.write(args.output, sr, (data_out * 32768).astype(np.int16))


# Run the main function if called from the command line
if __name__ == "__main__":
    import sys
    main(sys.argv)
