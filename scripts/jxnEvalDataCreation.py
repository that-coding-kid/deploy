from datetime import datetime, timedelta
from moviepy.video.io import ffmpeg_tools
import subprocess


def subtract_timestamps(timestamp1, timestamp2):
    time_format = "%H:%M:%S"
    dt1 = datetime.strptime(timestamp1, time_format)
    dt2 = datetime.strptime(timestamp2, time_format)
    time_difference = dt1 - dt2
    total_seconds = time_difference.total_seconds()
    return total_seconds



def mainFunc(videoSource,cycle,finalpath):


    offSetConstant = 1
    start_time_stamp = 0
    i = 0
    lastBSIndex = videoSource.rfind("\\")
    videoName = videoSource[lastBSIndex+1:]
    videoName = videoName[-4:]
    
    # print("HERE AT INSIDE FILE2 \n\n\n" + videoSource)
    print("Len Cycle: ",len(cycle))
    while (True):
        index = i%len(cycle)
        slice_end = start_time_stamp+(cycle[index]/offSetConstant)
        if (slice_end>1290):
            break
        clip_name = finalpath+"\clip"+str(i)+".mp4"
        ffmpeg_tools.ffmpeg_extract_subclip(videoSource,start_time_stamp,slice_end,clip_name)
        i+=1
        start_time_stamp = slice_end


