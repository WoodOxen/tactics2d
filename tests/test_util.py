import sys
sys.path.append(".")
sys.path.append("..")
from tactics2d.utils.preprocess import DLPPreprocess

def test_dlp_preprocess():
    source_path = "../tactics2d/data/trajectory_sample/DLP"
    target_path = "../tactics2d/data/trajectory_test_processed/DLP"
    file_id = 12

    processor = DLPPreprocess()
    processor.load(file_id, source_path)
    df_tracks = processor.process_tracks()
    df_tracks.to_csv(f"{target_path}/{file_id}.csv", index=False)


if __name__ == "__main__":
    test_dlp_preprocess()