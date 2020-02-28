
import pandas as pd
import os


def create_sequence_optical_flow_metadata_file(src_metadata_file_path):
    data_df = pd.read_csv(src_metadata_file_path)

    # Rename: image2 -> image3, image3 -> image5
    data_df = data_df.rename(columns={"image2": "image3", "image3": "image5"})

    # Create: image2 as <sequence_opticalflowGF_1.png>, and image4 as <sequence__opticalflowGF_2.png>
    data_df["image2"] = data_df["sequence"].astype(str) + "_opticalflowGF_1.png"
    data_df["image4"] = data_df["sequence"].astype(str) + "_opticalflowGF_2.png"
    data_df_selected = data_df[["sequence", "image1", "image2", "image3", "image4", "image5", "mask_MOG2", "has_animal"]]

    # Save the new metadata file.
    dest_metadata_file_path = "-opticalflowsequence".join(os.path.splitext(src_metadata_file_path))
    data_df_selected.to_csv(dest_metadata_file_path, index=False)
    print("Optical Flow Sequence metadata file has been written to %s" % dest_metadata_file_path)


def create_optical_flow_metadata_file(src_metadata_file_path):
    data_df = pd.read_csv(src_metadata_file_path)

    # Create: image2 as <sequence_opticalflowGF_1.png>, and image4 as <sequence__opticalflowGF_2.png>
    data_df["opticalflowGF_1"] = data_df["sequence"].astype(str) + "_opticalflowGF_1.png"
    data_df["opticalflowGF_2"] = data_df["sequence"].astype(str) + "_opticalflowGF_2.png"
    data_df["opticalflowGF_average"] = data_df["sequence"].astype(str) + "_opticalflowGF_average.png"

    # Save the new metadata file.
    dest_metadata_file_path = "-updated_opticalflow".join(os.path.splitext(src_metadata_file_path))
    data_df.to_csv(dest_metadata_file_path, index=False)
    print("Optical Flow Sequence metadata file has been written to %s" % dest_metadata_file_path)

if __name__ == "__main__":
    src_metadata_files = [
        "../data/final_dataset_train_balanced.csv",
        "../data/final_dataset_val_balanced.csv",
        "../data/final_dataset_test_balanced-shuffled.csv",
        "../data/final_dataset_train-trial.csv",
        "../data/final_dataset_val-trial.csv"
    ]
    for src_metadata_file in src_metadata_files:
        # create_sequence_optical_flow_metadata_file(src_metadata_file)
        create_optical_flow_metadata_file(src_metadata_file)
