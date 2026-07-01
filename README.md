# Retracing and Restoring: Chronological Context Preservation for Effective Dynamic Recommendation
<img width="1764" height="518" alt="image" src="https://github.com/user-attachments/assets/fe751539-f95b-4a53-b453-a082e5cfe3e9" />

This repository provides a reference implementation of TraceRrec as described in the following paper "[Retracing and Restoring: Chronological Context Preservation for Effective Dynamic Recommendation](https://doi.org/10.1145/3774904.3792407)", published at ACM Web Conference 2026 (full paper).

## Inputs
The structure of the input dataset is the following: 

```| user_id | item_id | timestamp | features |```.

The dataset is preprocessed through ```libaray_data.py```.

## Outputs
The model and accuracy of the best epoch are saved in the best_models folder and log folder, respectively.

## Usage
To run TraceRec on different datasets, use the following commands:
+ For LastFM:
```
python tracerec.py --dataset=lastfm --num_path_u=15 --num_path_i=15 --aggregation_method=GRU --seed=0 --gpu=1 --project
```

+ For MOOC:
```
python tracerec.py --dataset=mooc --num_path_u=35 --num_path_i=20 --aggregation_method=lstm --seed=0 --gpu=1 --project 
```

+ For Wikipedia:
```
python tracerec.py --dataset=wikipedia --num_path_u=10 --num_path_i=45 --aggregation_method=concat --seed=0 --gpu=1 --project 
```

+ For Yoochoose:
```
python tracerec.py --dataset=yoochoosebuy --num_path_u=20 --num_path_i=5 --aggregation_method=lstm --seed=0 --gpu=1 --project 
```

+ For Douban Movie:
```
python tracerec.py --dataset=douban_movie --num_path_u=20 --num_path_i=10 --aggregation_method=lstm --seed=0 --gpu=1 --project
```

+ For Amazon-Video:
```
python tracerec.py --dataset=video --num_path_u=20 --num_path_i=15 --aggregation_method=lstm --seed=0 --gpu=1 --project 
```
