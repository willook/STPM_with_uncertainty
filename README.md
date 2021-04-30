### Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection implementation (unofficial)
Unofficial pytorch implementation of  
Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection (STPM)  
\- Guodong Wang, Shumin Han, Errui Ding, Di Huang  (2021)  
https://arxiv.org/abs/2103.04257v2  

notice : This code is not official code and not verified yet. 

### Usage 
~~~
python train.py --phase 'train or test' --dataset_path '...\mvtec_anomaly_detection\bottle' --project_path 'path\to\save\results'
~~~

### MVTecAD ROC-AUC score (mean of n trials)
| Category | Paper (pixel-level) | This (pixel-level) | Paper (image-level) | This (image-level) |
| :-----: | :-: | :-: | :-: | :-: |
| carpet | 0.988 | 0.985(1)| - | 0.960(1) |
| grid | 0.990 | 0.989(1)| - | 0.995(1)|
| leather | 0.993 | 0.988(1)| - | 0.994(1) |
| tile | 0.974 | 0.937(1)| - | 0.978(1) |
| wood | 0.972 | 0.812(1)| - | 0.886(1) |
| bottle | 0.988 | 0.968(1)| - | 0.997(1) |
| cable | 0.955 | 0.718(1)| - | 0.779(1) |
| capsule | 0.983 | 0.975(1)| - | 0.931(1) |
| hazelnut | 0.985 | 0.941(1)| - | 0.812(1) |
| metal nut | 0.976 | 0.967(1)| - | 0.999(1) |
| pill | 0.978 | 0.948(1)| - | 0.784(1) |
| screw | 0.983 | 0.984(1)| - | 0.869(1) |
| toothbrush | 0.989 | 0.980(1) | - | 0.878(1) |
| transistor | 0.825 | 0.569(1)| - | 0.415(1) |
| zipper | 0.985 | 0.979(1)| - | 0.942(1) |
| mean | 0.970 | 0.916(1) | 0.955 | 0.881(1) |

Under test.    

### Localization results   


![plot](./samples/bent_003_arr.png)
![plot](./samples/bent_009_arr.png)
![plot](./samples/broken_000_arr.png)
![plot](./samples/metal_contamination_003_arr.png)
![plot](./samples/thread_001_arr.png)
![plot](./samples/thread_005_arr.png)