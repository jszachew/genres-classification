#Music Genre Classification (with text representation)

This project is to investigate music genre classification problem when songs are in its texts representation.

Dataset used here is based on Million Songs Dataset (http://www.millionsongdataset.com/) with its additional sets (http://www.millionsongdataset.com/pages/additional-datasets/):

- https://www.tagtraum.com/msd_genre_datasets.html (genre labels)
- http://www.millionsongdataset.com/musixmatch/ (lyrics)

As tagtraum dataset provides two approaches of genre labeling, both of them were analysed (they are called CD1 and CD2)

## Analysed models

There were four approaches investigated:

- Random forest
- Logistic regression
- CNN
- MLP

## Results

Best results was achieved using MLP network with SMOTE algorithm applied (CD1 as a dataset labeling). 
Confusion matrix for that approach looks like:

![Alt text](results/final_results/CD1_MLP_dropout_200/CD1_MLP_with_dropout_sgd_network_2022-01-09_17-22-35.png?raw=true "MLP confusion matrix")
More precise metrics are below:

|                 | Precyzja | Czułość |       F1 | Ilość  |
| --------------- | -------- | ------- | -------- |--------|
|      Blues      |   0.97   |   1.00  |    0.99  | 1234   |
|      Country    |   0.86   |   0.86  |    0.86  | 1233   |
|   Electronic    |   0.81   |   0.83  |    0.82  | 1234   |
|         Folk    |   0.93   |   0.92  |    0.93  | 1233   |
|International    |   0.98   |   0.95  |    0.97  | 1233   |
|         Jazz    |   0.98   |   0.96  |    0.97  | 1233   |
|        Latin    |   0.98   |   0.97  |    0.98  | 1233   |
|      New Age    |   0.91   |   1.00  |    0.95  | 1234   |
|     Pop\_Rock    |   0.62   |   0.57  |    0.59  | 1233   |
|          Rap    |   0.95   |   0.90  |    0.93  | 1233   |
|       Reggae    |   0.97   |   0.96  |    0.97  | 1234   |
|          RnB    |   0.87   |   0.92  |    0.89  | 1234   |
|        Vocal    |   1.00   |   0.99  |    0.99  | 1234   | 
|     accuracy    |          |         |    0.91  | 16035  |
|    macro avg    |   0.91   |   0.91  |    0.91  | 16035  |
| weighted avg    |   0.91   |   0.91  |    0.91  | 16035  |
Also for CD2 labeling results looks pretty good:
![Alt text](results/final_results/CD2_MLP_200/CD2_MLP_with_dropout_sgd_network_2022-01-15_21-16-04.png?raw=true "MLP confusion matrix")

And metrics for that are:

|                 | Precyzja | Czułość |       F1 |  Ilość |
| --------------- | -------- | ------- | -------- |--------|
|       Blues     |    0.93  |   0.96  |    0.95  |   2730 | 
|     Country     |    0.75  |   0.83  |    0.79  |   2730 | 
|  Electronic     |    0.78  |   0.71  |    0.74  |   2730 | 
|        Folk     |    0.89  |   0.87  |    0.88  |   2729 | 
|        Jazz     |    0.93  |   0.93  |    0.93  |   2730 | 
|       Latin     |    0.92  |   0.96  |    0.94  |   2729 | 
|       Metal     |    0.68  |   0.75  |    0.71  |   2730 | 
|     New Age     |    0.93  |   1.00  |    0.96  |   2730 | 
|         Pop     |    0.56  |   0.29  |    0.39  |   2729 | 
|        Punk     |    0.80  |   0.84  |    0.82  |   2730 | 
|         Rap     |    0.90  |   0.92  |    0.91  |   2730 | 
|      Reggae     |    0.96  |   0.92  |    0.94  |   2730 | 
|         RnB     |    0.74  |   0.82  |    0.78  |   2730 | 
|        Rock     |    0.36  |   0.39  |    0.38  |   2730 | 
|       World     |    0.97  |   0.95  |    0.96  |   2730 |
|    accuracy     |          |         |    0.81  |  40947 | 
|   macro avg     |    0.81  |   0.81  |    0.80  |  40947 | 
|weighted avg     |    0.81  |   0.81  |    0.80  |  40947 | 


