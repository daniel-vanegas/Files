# Generate Dataset

##  Generate .npz files from the raw dataset. 

The script will search for all runs in the dataset/data folder and generate a .npz file next to each run folder.

Default root, all runs:
```bash
python3 generate_npz_data.py
```

One scenario, relative:
```bash
python3 generate_npz_data.py E01
```

One run, relative:
```bash
python3 generate_npz_data.py E01/run_20260928_073644_nav2_dwb_abstract
```

Absolute path:
```bash
python3 generate_npz_data.py /path/to/dataset/data/E01
```

Example output

```bash
data root: /home/user/mapless/dataset/data
target:    /home/user/mapless/dataset/data
runs:      2
[E01/run_20260928_073644_nav2_dwb_abstract]
  wrote /home/user/mapless/dataset/data/E01/run_20260928_073644_nav2_dwb_abstract.npz (383 samples)
[E01/run_20260928_091205_nav2_dwb_abstract]
  wrote /home/user/mapless/dataset/data/E01/run_20260928_091205_nav2_dwb_abstract.npz (271 samples)
done: 2/2
```

### Notes

- Episodes are not filtered here. Collision and stuck policies are applied downstream in dataset_utils.filter_by_episode.

- The .npz stores an object array of dicts. Loading requires np.load(path, allow_pickle=True).

- Rerunning the script overwrites existing .npz files next to each run folder.