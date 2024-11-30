from sklearn.preprocessing import StandardScaler

def preprocess_subjects(subjects, split_name):
    print(f"Preprocessing {split_name} subjects for fold: {fold_dir}")
    all_windows = []
    all_labels = []

    for subject in subjects:
        subject_dir = os.path.join(raw_data_dir, subject)
        if not os.path.exists(subject_dir):
            print(f"Warning: Subject directory {subject_dir} does not exist. Skipping...")
            continue

        mat_files = [f for f in os.listdir(subject_dir) if f.endswith('.mat')]
        if not mat_files:
            print(f"Warning: No .mat files found in {subject_dir}. Skipping...")
            continue

        for mat_file in mat_files:
            file_path = os.path.join(subject_dir, mat_file)
            try:
                emg_data, labels = load_fold_data(file_path, fold_idx=1)

                # Apply preprocessing steps
                emg_data, labels = clean_data(emg_data, labels)

                # Normalize the data: Separate training and test normalization
                # Create a train-test split or fold processing here
                if split_name == "train":
                    # For training data, fit the scaler and normalize
                    scaler = StandardScaler()
                    emg_data = scaler.fit_transform(emg_data)
                else:
                    # For validation data, normalize using the same scaler
                    emg_data = scaler.transform(emg_data)  # Use the same scaler for test data

                windows, window_labels = create_sliding_windows(emg_data, labels)

                all_windows.append(windows)
                all_labels.append(window_labels)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

    # After processing all subjects, combine the windows and labels
    if all_windows:
        all_windows = np.concatenate(all_windows, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        positional_enc = positional_encoding(all_windows.shape[1], all_windows.shape[2])
        all_windows += positional_enc

        # Save the processed data
        save_path = os.path.join(processed_dir, f"{split_name}_data.pkl")
        joblib.dump((all_windows, all_labels), save_path)
        print(f"Saved {split_name} data to {save_path}")
    else:
        print(f"No data to save for {split_name} in fold {fold_dir}")
