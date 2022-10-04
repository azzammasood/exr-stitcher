import glob
import os
import argparse
import numpy as np
import re
import OpenEXR as openexr
import Imath
from memory_profiler import profile
import operator
from natsort import natsorted

_np_to_exr = {
    np.float16: Imath.PixelType.HALF,
    np.float32: Imath.PixelType.FLOAT,
    np.uint32: Imath.PixelType.UINT,
}
_exr_to_np = dict(zip(_np_to_exr.values(), _np_to_exr.keys()))
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def main():
    # Add command line arguments
    args = add_arguments('--input_folder', '--output_folder')

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

    # Sort the list containing filenames, removing out extra files (files to paste) into another list 'extras'
    sorted_main_files = []
    extras = []
    for filename in sorted_files:
        if len(re.findall(r'\d+', filename)) < 3:
            sorted_main_files.append(filename)
        if len(re.findall(r'\d+', filename)) > 2:
            if int(re.findall(r'\d+', filename)[-1]) < 2:
                sorted_main_files.append(filename)
            else:
                extras.append(filename)

    print('\nMain files:', sorted_main_files)
    print('Extra files:', extras)

    # Get size of grid
    grid_size = get_grid_size(all_files)
    grid_size = [int(x) for x in grid_size]
    print('\nGrid size:', tuple(grid_size), '\n')

    # Check incomplete list of files
    print("Check 1: Checking for missing files for completion of grid..")
    assert len(sorted_main_files) % 2 == 0, "Incomplete images for requested grid"
    print("Check successful! \n")

    # Reading the exr files in a list
    print("Check 2: EXR files must be complete in accordance with their pixels")
    exr_files_list, channel_names = [], []
    for file in sorted_main_files:
        exr_file, channels = read_exr(file)
        exr_files_list.append(exr_file)
        channel_names.append(channels)
    print("Check successful!\n")

    # Sometimes Photoshop saves images with channel name Background.R instead of R. Same goes for other channels.
    channel_names = fix_channel_names(channel_names, reverse=False)

    # Split to get grid
    grid = np.array_split(exr_files_list, grid_size[0])

    # Read the additional images that need to be pasted inside cells
    exr_files_to_paste, channel_names_paste = [], []
    positions = []
    if len(extras) > 0:
        print("Bonus Check: Extra images must be complete in accordance with their pixels")
        for file in extras:
            exr_file_to_paste, channels = read_exr(file)
            exr_files_to_paste.append(exr_file_to_paste)
            channel_names_paste.append(channels)
            positions.append(re.findall(r'\d+', file))

        # Confirm if their dimensions are smaller or equal to the cell they need to be pasted into
        positions = convert_to_int(positions)
        check_smaller_resolution(exr_files_to_paste, positions, grid)

        # Confirm if their channels are the same as the previous images
        channel_names_paste = fix_channel_names(channel_names_paste)
        assert channel_names_paste[0] == channel_names[0], "All images must have same channels"

    # Change current working directory to output folder
    os.chdir('../' + args.output_folder)

    # Check if all images have same channel configuration
    check_channels_similarity(exr_files_list)

    # Check if rows have same height and columns have same width
    window_height, window_width = check_heights_and_widths(exr_files_list, grid_size)

    # Write separate output file for each image
    display_window = (window_height, window_width)
    write_exr('Stitched_Image.exr', grid, exr_files_to_paste, positions, display_window, channel_names[0])

# @profile
def write_exr(filename, rows_list, extra_images, positions_of_extra_images, display_window, channel_names):
    """Writes the values in a multi-channel ndarray into an EXR file.

    Creates a header with the resolution of the complete stitched image.
    For each row in rows_list (which is the list of main images), concatenate
    individual rows of data of each image in this row, to form a single scanline and
    write out this single scanline onto the output image. This process is repeated for
    all rows of data. Next, if there are any extra images, place them in the concatenated
    scanline at the appropriate indices. Then write out this new scanline. Close the exr
    file after completion of writing.

  Args:
    filename: The filename of the output file
    rows_list: The list of rows of images, so basically a list of lists of lists
    extra_images: The list of extra images to be pasted
    positions_of_extra_images: The list of lists of rows by columns of each extra image
    display_window: The total resolution of the stitched image
    channel_names: A list of strings with length = channels

  Raises:
    TypeError: If the numpy array has an unsupported type.
    ValueError: If the length of the array and the length of the channel names
      list do not match.
  """
    header = openexr.Header(display_window[1], display_window[0])
    try:
        exr_channel_type = Imath.PixelType(_np_to_exr[rows_list[0][0].dtype.type])
    except KeyError:
        raise TypeError('Unsupported numpy type: %s' % str(rows_list[0][0].dtype))
    header['channels'] = {
        n: Imath.Channel(exr_channel_type) for n in channel_names
    }

    exr = openexr.OutputFile(filename, header)

    for grid_row_number, row in enumerate(rows_list):    # Get one grid-row of images
        combined = []
        for data_row in range(row[0].shape[0]):  # For a single row
            for image_number in range(len(row)): # For all images in this row
                two_d_row = row[image_number][data_row, :, :]  # Get one row from this image
                three_d_row = two_d_row[None, :, :]   # Convert it to 3D
                if image_number == 0:
                    combined = three_d_row
                else:
                    combined = np.concatenate((combined, three_d_row), axis=1)   # Concatenate it

            # Consider extra images
            if len(extra_images) > 0:   # If there are extra images
                for i in range(len(extra_images)):  # For an extra image
                    if grid_row_number == positions_of_extra_images[i][0] - 1:  # If grid row is equal to row number of extra image
                        if data_row < extra_images[i].shape[0]: # Until the row count of image to paste is less than the row counts of image already in cell
                            sum_widths = 0
                            images_to_add_to_offset = positions_of_extra_images[i][1]   # Get column of extra image
                            images_to_add_to_offset -= 1    # Since grid starts from 0th row and 0th column
                            previous_images = row[:images_to_add_to_offset]

                            # Get row of data from extra image
                            two_d_image_to_paste = extra_images[i][data_row, :, :]  # This will result in 2D array, since a single row is not counted in the 3D dimensions
                            three_d_image_to_paste = two_d_image_to_paste[None, :, :]   # Convert it into 3D array for broadcasting to work

                            if len(previous_images) > 0:    # If offset needs to be generated
                                for img in previous_images:
                                    if images_to_add_to_offset == 0:
                                        sum_widths = 0  # First image in list
                                    else:
                                        sum_widths += img.shape[0]  # Sum the width of each image obtain offset
                                offset = sum_widths + extra_images[i].shape[1]
                                combined[:, sum_widths:offset, :] = three_d_image_to_paste
                            else:
                                combined[:, :extra_images[i].shape[1], :] = three_d_image_to_paste


            channel_data = [combined[..., i] for i in range(combined.shape[-1])]  # List of lists, each sublist containing channel data

            scanline = dict((n, d.tobytes()) for n, d in zip(channel_names, channel_data))
            exr.writePixels(scanline, 1)
        print('Writing of grid row', grid_row_number+1, 'successful.')
    print("\n")
    exr.close()

def read_exr(filename, channel_names=None):
    """ Opens an EXR file and copies the requested channels into an ndarray.

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
    # Stack the arrays so that the channels dimension is the last (fastest changing) dimension.
    return np.stack(channels, axis=-1)


def match_filenames(file_extension, regular_expression):
    """
    Uses a regular expression to match filenames using the re module
    and glob module.

    Args:
        file_extension: The extension of the files to be searched
        regular_expression: The regular expression to be matched

    Returns:
        A list of matched filenames

    """
    return [file for file in glob.glob('*' + file_extension)
            if re.match(regular_expression, file)]


def check_channels_similarity(list_of_images):
    """
    Compare all channels from all images in the  supplied list.
    Append the third dimension of each image in a list, then
    convert this list into a set to remove duplicated. If there
    is one element remaining in the set, that means all images
    have same number of channels.

    Args:
        list_of_images: A list of 3D arrays

    Raises:
        AssertionError: If length of the set is not 1, meaning images don't
                        have same number of channels
    """
    channels = []
    for image in list_of_images:
        channels.append(image.shape[-1])
    assert len(set(channels)) == 1, "All images must have same channels"

def check_heights_and_widths(exr_files_list, grid_size):
    """
    Compare the heights of all images belonging to the same row,
    and the widths of all images belonging to the same column.
    Also sums the heights of all images in a column,
    and sums the widths of all images in a row, to get the
    resolution of the final stitched image. This is used as
    our displayWindow when creating the header for our output
    exr file.

    Args:
        exr_files_list: The list of 3D arrays for each exr file
        grid_size: The tuple of rows by columns

    Returns:
        window_height: The height of the output image
        window_width: The width of the output image

    Raises:
        AssertionError: If images in a row do not have same heights,
                        or if images in a column do not have same widths.

    """
    columns = exr_files_list[0::grid_size[1]]
    width_of_columns = [exr.shape[1] for exr in columns]

    print("\nCheck 3: Equal widths of images belonging to the same column..")

    assert len(set(width_of_columns)) == 1, "Widths of images in same column must be same"

    print("Check successful! \n")

    window_height = sum([exr.shape[0] for exr in columns])

    print("Check 4: Equal heights of images belonging to the same row..")
    for i in range(0, len(exr_files_list), grid_size[1]):
        row = exr_files_list[i:i+grid_size[1]]
        heights = []
        widths = []
        for j in row:
            heights.append(str(j.shape[0])+' px')
            widths.append(j.shape[1])
        assert len(set(heights)) == 1, "Heights of images in same row must be same"
    print("Check successful! \n")

    window_width = sum(widths)

    return window_height, window_width

def add_arguments(name_arg1, name_arg2):
    """
    Add command-line arguments using the ArgumentParser module.

    Args:
        name_arg1: string of name of argument 1
        name_arg2: string of name of argument 2

    Returns:
        args: the parsed arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(name_arg1, type=str, help='Directory where input images are located')
    parser.add_argument(name_arg2, type=str, help='Directory where output image should be written')
    args = parser.parse_args()
    return args


def get_grid_size(all_files):
    """
    Calculates the grid size (rows by columns) from the
    given list of string filenames.

    Args:
        all_files: list of strings

    Returns:
        grid_size: a tuple of rows by columns

    """

    grid_cells = []
    for filename in all_files:
        grid_cells.append(re.findall(r'\d+', filename))

    num_rows = [row[0] for row in grid_cells]
    num_columns = [column[1] for column in grid_cells]
    grid_size = (max(num_rows), max(num_columns))

    return grid_size

def convert_to_int(list):
    """ Converts elements to integers

    Each element in a list of elements is converted to an integer
    and stored in a new list.

    Args:
        list: A list of elements
    Returns:
        list with all integerments
    """
    return [[int(num) for num in lstt[:]] for lstt in list]

def check_smaller_resolution(exr_files_to_paste, positions, grid):
    """ Compares resolutions of extra files with the cells they need to be pasted into.

    If the image to be pasted has greater or equal resolution than the cell it needs
    to be pasted into, then an AssertionError is raised, otherwise the program continues
    execution.

    Args:
          exr_files_to_paste: list of extra images to be pasted
          positions: list of grid and columns of the extra images
          grid: tuple of grid size (rows by columns)

    Returns:
          None

    Raises:
        AssertionError: If the image to be pasted has greater or equal resolution than the cell
    """
    for i in range(len(exr_files_to_paste)):
        row = positions[i][0]
        column = positions[i][1]
        target_cell = grid[row-1][column-1]
        target_cell_width = target_cell.shape[1]
        target_cell_height = target_cell.shape[0]
        paste_image_width = exr_files_to_paste[i].shape[1]
        paste_image_height = exr_files_to_paste[i].shape[0]
        assert paste_image_width < target_cell_width and paste_image_height < target_cell_height, "Image to be pasted in must have smaller dimensions than that of cell"
    print("Check successful!")

def fix_channel_names(list_of_channels, reverse=True):
    """
    Fixes channel naming done in Photoshop.
    Replaces the channel names in the list of channel names.
    If the channel_name is 'Background.R' instead of 'R', it
    replaces it with 'R'.

    Args:
        list_of_channels: The list of lists of strings of channel names
    Returns:
        list_of_channelsL: The fixed list
    """
    for item in range(len(list_of_channels)):
        for i, j in enumerate(list_of_channels[item]):
            if j == 'Background.R':  # Photoshop sometimes saves EXRs with these channel names
                list_of_channels[item][i] = 'R'
            elif j == 'Background.G':
                list_of_channels[item][i] = 'G'
            elif j == 'Background.B':
                list_of_channels[item][i] = 'B'
            elif j == 'Background.A':
                list_of_channels[item][i] = 'A'
        if reverse == True:
            list_of_channels[item].reverse()

    return list_of_channels

if __name__ == '__main__':
    main()
