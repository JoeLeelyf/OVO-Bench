import io
import base64
from moviepy.editor import VideoFileClip
import numpy as np
import tempfile
from PIL import Image

"""
    For offline models that require a whole video as input, such as Gemini
"""
def process_video_to_base64(video_path, end_time, start_time=0):
    video = VideoFileClip(video_path)
    duration = video.duration
    try:
        end_time = min(end_time, duration)
        clip = video.subclip(start_time, end_time)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file_path = temp_file.name
        
        clip.write_videofile(temp_file_path, codec="libx264", fps=clip.fps)
        
        with open(temp_file_path, 'rb') as video_file:
            base64_encoded = base64.b64encode(video_file.read())
            base64_string = base64_encoded.decode('utf-8')
        
    finally:
        # 关闭视频对象
        video.close()
        # 删除临时文件
        if temp_file_path:
            import os
            os.remove(temp_file_path)
    
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
    # Set endpoint=True to ensure that last frame are included in the sampling frames
    sampled_times = np.linspace(0, end_time, max_frames, endpoint=True)
    sampled_frames = np.array([clip.get_frame(t) for t in sampled_times])
    video.close()
    return sampled_frames

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def split_save_frames(video_path, end_time, start_time=0, max_frames=64):
    video = VideoFileClip(video_path)
    
    total_frames = video.fps * (end_time - start_time)
    if total_frames < max_frames:
        frames = [frame for frame in video.iter_frames()]
    else:
        clip = video.subclip(start_time, end_time)
        # Set endpoint=True to ensure that last frame are included in the sampling frames
        sampled_times = np.linspace(0, end_time - start_time, max_frames, endpoint=True)
        frames = [clip.get_frame(t) for t in sampled_times]
        clip.close()
    
    frame_paths = []
    for i, frame in enumerate(frames):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        frame_image = Image.fromarray(frame)
        frame_image.save(temp_file.name)
        frame_paths.append(temp_file.name)
    
    video.close()
    return frame_paths