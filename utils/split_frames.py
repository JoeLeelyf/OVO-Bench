import io
import base64
from moviepy.editor import VideoFileClip
import numpy as np
import tempfile

"""
    For offline models that require a whole video as input, such as Gemini
"""
def split_videos(video_path, end_time, start_time=0):
    video = VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    clip.write_videofile(temp_file.name, codec="libx264", fps=clip.fps)
    video.close()
    return temp_file.name

def encode_video(video_path):
    with open(video_path, 'rb') as video_file:
        video_data = video_file.read()
        
    base64_encoded = base64.b64encode(video_data)
    base64_string = base64_encoded.decode('utf-8')
        
    return base64_string

"""
    For offline models that require frames as input, such as GPT
"""
def split_frames(video_path, end_time, start_time=0, max_frames = 64):
    video = VideoFileClip(video_path)
    # If frames are less than max_frames, return all frames as np array
    if video.fps * (end_time - start_time) < max_frames:
        return np.array([frame for frame in video.iter_frames()])
    
    # Else, evenly sample and ensure to contain last frame
    clip = video.subclip(start_time, end_time)
    sampled_times = np.linspace(0, end_time, max_frames, endpoint=True)
    sampled_frames = np.array([clip.get_frame(t) for t in sampled_times])
    video.close()
    return sampled_frames

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')