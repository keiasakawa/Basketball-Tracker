from utils import read_video, save_video
from trackers import Tracker
import cv2

def main():
    # Read Video
    video_frames = read_video('input_videos/nba2.mp4')

    tracker = Tracker('models/best (2).pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True,stub_path='stubs/track_stubs.pkl')

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if  __name__ == '__main__':
    main()