# change frame rate with ffmpeg
""" Run locally directly using ffmpeg command to reduce required time """
import cv2
import glob
import os
import pandas
import subprocess
# Change these to fit your specific directory configuration
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
input_path = r"D:\Carleton\AML\Project\UCFRawVideos"
output_path = r"D:\Carleton\AML\Project\UCF101"
video_file_extension = "*" # optional video file extension

# Get all video files in the input directory
video_filenames = glob.glob(os.path.join(input_path, "*." + video_file_extension))

# iterate over all videos and change FPS with ffmpeg
for name in video_filenames:    
    basename = os.path.splitext(os.path.basename(name))[0]
    output_subdirectory = os.path.join(output_path, basename)
    
    os.makedirs(output_subdirectory)
    output_filename = os.path.join(output_path, basename, basename + ".avi")
    command = ffmpeg_path + " -i " + name + " -filter:v fps=fps=6 " + output_filename
    print(command)

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    print(stderr)