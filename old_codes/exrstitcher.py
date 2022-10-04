# Stitches exrs using numpy, then writes out stitched image
import glob
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import numpy as np
import re
import OpenEXR as openexr
import Imath
from PIL import Image
import cv2
import inspect
import psutil
from memory_profiler import memory_usage
from memory_profiler import profile
from natsort import natsorted

_np_to_exr = {
    np.float16: Imath.PixelType.HALF,
    np.float32: Imath.PixelType.FLOAT,
    np.uint32: Imath.PixelType.UINT,
}
_exr_to_np = dict(zip(_np_to_exr.values(), _np_to_exr.keys()))


def main():
    # Add and read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='Directory where input images are located')
    parser.add_argument('--output_folder', type=str, help='Directory where output image should be written')
    args = parser.parse_args()

    # Changing current working directory to input folder
    os.chdir(args.input_folder)

    # Get names of required files
    files_first_convention = match_filenames('.exr', r'^u(?!0)\d{1,2}?_v(?!0)\d{1,2}?\.exr$')  # ^ matches start of string.
                                                                                       # $ matches start of string.
                                                                                       # u(?!0)\d? will match 'u' only if it's not followed by the digit 0. The ? matches 0 or 1 repitions of the preceding digit.
    files_first_convention = np.array(files_first_convention)
    files_second_convention = match_filenames('.exr', r'^u(?!0)\d{1,2}?_v(?!0)\d{1,2}?_(?!0)\d{1,2}?\.exr$')
    files_second_convention = np.array(files_second_convention)
    all_files = np.concatenate((files_first_convention, files_second_convention))
    sorted_files = natsorted(all_files)

    grid_cells = []
    for filename in all_files:
        grid_cells.append(re.findall(r'\d+', filename))

    # Sort the list containing filenames, removing out files that need to be pasted into a grid cell, into another list
    sorted_all_files = []
    extras = []
    for filename in sorted_files:
        if len(re.findall(r'\d+', filename)) < 3:
            sorted_all_files.append(filename)
        if len(re.findall(r'\d+', filename)) > 2:
            if int(re.findall(r'\d+', filename)[-1]) < 2:
                sorted_all_files.append(filename)
            else:
                extras.append(filename)

    rows = [row[0] for row in grid_cells]
    columns = [column[1] for column in grid_cells]

    # Check incomplete list of files
    assert len(sorted_all_files) % 2 == 0, "Incomplete images for requested grid"

    # Reading the exr files in a list
    exr_files_list = []
    channel_names = []
    # Reading the exr files in a list
    for file in sorted_all_files:
        exr_file, channels = read_exr(file)
        exr_files_list.append(exr_file)
        channel_names.append(channels)

    # Calculate grid size
    grid_size = (max(rows), max(columns))

    # Check if all images have same channel configuration
    check_channels_similarity(exr_files_list)

    # Check if rows have same height and columns have same width
    window_height, window_width = check_heights_and_widths(exr_files_list, grid_size)

    # Get rows of images
    rows = np.array_split(exr_files_list, int(grid_size[0]))

    # Read the additional images that need to be pasted inside cells
    exr_files_to_paste = []
    channel_names_paste = []
    positions = []
    if len(extras) > 0:
        for file in extras:
            exr_file_to_paste, channels = read_exr(file)
            exr_files_to_paste.append(exr_file_to_paste)
            channel_names_paste.append(channels)
            positions.append(re.findall(r'\d+', file))

        # Confirm if their channels are the same as the previous images
        for item in range(len(channel_names_paste)):
            for i, j in enumerate(channel_names_paste[item]):
                if j == 'Background.R':
                    channel_names_paste[item][i] = 'R'
                elif j == 'Background.G':
                    channel_names_paste[item][i] = 'G'
                elif j == 'Background.B':
                    channel_names_paste[item][i] = 'B'
            channel_names_paste[item].reverse()

        assert channel_names_paste[0] == channel_names[0], "All images must have same channels"

        # Paste the additional images
        positions = [[int(x) for x in sublist] for sublist in positions]
        for i, image in enumerate(exr_files_to_paste):
            target_cell = rows[positions[i][0]-1][positions[i][1]-1]
            target_cell[:image.shape[0]-1, :image.shape[1]-1] = image[:image.shape[0]-1, :image.shape[1]-1]
            rows[positions[i][0] - 1][positions[i][1] - 1] = target_cell

    # Stitch images horizontally
    list_stitched_rows = []
    for row in rows:
        for img in range(len(row)):
            if img == 0:
                stitched_row = row[img]
            else:
                stitched_row = np.concatenate((stitched_row, row[img]), axis=1)
                result = stitched_row
        list_stitched_rows.append(stitched_row)

    if (len(rows) > 1):
        # Stitch rows vertically
        for row in range(len(list_stitched_rows)):
            if row == 0:
                stitched_vertically = list_stitched_rows[row]
            else:
                stitched_vertically = np.concatenate((stitched_vertically, list_stitched_rows[row]), axis=0)
                result = stitched_vertically


    # Change current working directory to output folder
    os.chdir('../' + args.output_folder)

    # Write the output file one scanline at a time
    write_exr(filename='Stitched.exr', image=result, channel_names=channel_names[0], number_of_scanlines=1)

def read_exr(filename, channel_names=None):
    """Opens an EXR file and copies the requested channels into an ndarray.

  The Python openexr wrapper uses a dictionary for the channel header, so the
  ordering of the channels in the underlying file is lost. If channel_names is
  not passed, this function orders the output channels with any present RGBA
  channels first, followed by the remaining channels in alphabetical order.
  By convention, RGBA channels are named 'R', 'G', 'B', 'A', so this function
  looks for those strings.

  Args:
    filename: The name of the EXR file.
    channel_names: A list of strings naming the channels to read. If None, all
      channels will be read.

  Returns:
    A numpy array containing the image data, and a list of the corresponding
      channel names.
  """
    exr = openexr.InputFile(filename)
    assert exr.isComplete() == True, filename + " is corrupt/incomplete"
    if channel_names is None:
        remaining_channel_names = list(exr.header()['channels'].keys())
        conventional_rgba_names = ['R', 'G', 'B', 'A']
        present_rgba_names = []
        # Pulls out any present RGBA names in RGBA order.
        for name in conventional_rgba_names:
            if name in remaining_channel_names:
                present_rgba_names.append(name)
                remaining_channel_names.remove(name)
        channel_names = present_rgba_names + sorted(remaining_channel_names)

    return np.array(channels_to_ndarray(exr, channel_names)), channel_names

# @profile
def write_exr(filename, image, channel_names, number_of_scanlines):
    """Writes the values in a multi-channel ndarray into an EXR file.

  Args:
    filename: The filename of the output file
    values: A numpy ndarray with shape [height, width, channels]
    channel_names: A list of strings with length = channels

  Raises:
    TypeError: If the numpy array has an unsupported type.
    ValueError: If the length of the array and the length of the channel names
      list do not match.
  """
    header = openexr.Header(image.shape[1], image.shape[0])
    try:
        exr_channel_type = Imath.PixelType(_np_to_exr[image.dtype.type])
    except KeyError:
        raise TypeError('Unsupported numpy type: %s' % str(image.dtype))
    header['channels'] = {
        n: Imath.Channel(exr_channel_type) for n in channel_names
    }

    exr = openexr.OutputFile(filename, header)

    channel_data = [image[..., i] for i in range(image.shape[-1])]  # List of lists, each sublist containing channel data

    for x, y, z in zip(*channel_data):
        one_scanline = [x, y, z]
        scanline = dict((n, d.tobytes()) for n, d in zip(channel_names, one_scanline))
        exr.writePixels(scanline, number_of_scanlines)

    exr.close()


def channels_to_ndarray(exr, channel_names):
    """Copies channels from an openexr.InputFile into a numpy array.

  If the EXR image is of size (width, height), the result will be a numpy array
  of shape (height, width, len(channel_names)), where the last dimension holds
  the channels in the order they were specified in channel_names. The requested
  channels must all have the same datatype.

  Args:
    exr: An openexr.InputFile that is already open.
    channel_names: A list of strings naming the channels to read.

  Returns:
    A numpy ndarray.

  Raises:
    ValueError: If the channels have different datatypes.
    RuntimeError: If a channel has an unknown type.
  """
    channels_header = exr.header()['channels']
    window = exr.header()['dataWindow']
    width = window.max.x - window.min.x + 1
    height = window.max.y - window.min.y + 1

    def read_channel(channel):
        """Reads a single channel from the EXR."""
        channel_type = channels_header[channel].type
        try:
            numpy_type = _exr_to_np[channel_type.v]
        except KeyError:
            raise RuntimeError('Unknown EXR channel type: %s' % str(channel_type))
        flat_buffer = np.frombuffer(exr.channel(channel), numpy_type)
        return np.reshape(flat_buffer, [height, width])

    channels = [read_channel(c) for c in channel_names]
    if any([channels[0].dtype != c.dtype for c in channels[1:]]):
        raise ValueError('Channels have mixed datatypes: %s' %
                         ', '.join([str(c.dtype) for c in channels]))
    # Stack the arrays so that the channels dimension is the last (fastest
    # changing) dimension.
    return np.stack(channels, axis=-1)


def match_filenames(file_extension, regular_expression):
    return [file for file in glob.glob('*' + file_extension)
            if re.match(regular_expression, file)]


def check_channels_similarity(list_of_images):
    channels = []
    for image in list_of_images:
        channels.append(image.shape[-1])
    assert len(set(channels)) == 1, "All images must have same channels"


def getnameofchannels(channels_list):
    if channels_list[0] == 1:
        return ['R']
    elif channels_list[0] == 2:
        return ['R', 'G']
    elif channels_list[0] == 3:
        return ['R', 'G', 'B']
    elif channels_list[0] == 4:
        return ['R', 'G', 'B', 'A']


def check_heights_and_widths(exr_files_list, grid_size):
    columns = exr_files_list[0::int(grid_size[1])]
    width_of_columns = [exr.shape[1] for exr in columns]
    assert len(set(width_of_columns)) == 1, "Widths of images in same column must be same"
    window_height = sum([exr.shape[0] for exr in columns])

    for i in range(0, len(exr_files_list), int(grid_size[0])):
        row = exr_files_list[i:i + 4]
        heights = []
        widths = []
        for j in row:
            heights.append(j.shape[0])
            assert len(set(heights)) == 1, "Heights of images in same row must be same"
            widths.append(j.shape[1])

    window_width = sum(widths)
    return window_height, window_width

if __name__ == '__main__':
    main()
