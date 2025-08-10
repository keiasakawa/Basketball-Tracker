from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    # Read Video
    video_frames = read_video('input_videos/nba2.mp4')

    tracker = Tracker('models/best (2).pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True,stub_path='stubs/track_stubs.pkl')

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['Player'][0])

    print(team_assigner.team_colors)

    for frame_num,  player_track in enumerate(tracks['Player']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['Player'][frame_num][player_id]['team'] = team
            tracks['Player'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')

    # player = tracks['Player'][150][3]
    # bbox = player['bbox']
    # frame = video_frames[0]

    # cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    # cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)

if  __name__ == '__main__':
    main()