
## Public opinion monitoring through collective semantic analysis of tweets  

</br>


### Overview

The goal of this project was to monitor the public opinion regarding presidential elections. For this purpose,
we chose Twitter as our data source and used deep neural networks to examine tweets semantically.
We actually extract four different semantic entities: polarity, offensiveness, figurative language, bias.
This is achieved by the respective four neural classifiers after training them with properly annotated tweet datasets.

</br>

The **architectures** we employed were found in:

1. https://github.com/efpm04013/finalexp34 , which incorporates a parallel combination of CNN and Bi-LSTM.

2. https://github.com/DheerajKumar97/US-2020-Election-Campaign-Youtube-Comments-Sentiment-Analysis-RNN-Bidirect--lstm-Flask-Deployment , which incorporates a Bi-LSTM along with a stack of Fully-Connected layers.  
  
</br>

We name the four-element vector extracted for each tweet **descriptor**. After we get all the descriptors for the
selected tweet dataset (using the trained models), we aggregate them daily (mean, median) to create time series.  

</br>

Finally we apply **time series forecasting** to the extracted data. The model used to that end was found in:

1. https://github.com/HansikaPH/time-series-forecasting , which incorporates a simple LSTM.

The model produces forecasts for a **7-day horizon**, after being fed a 9-day input window of past observations/data.  

</br>

The mechanism explained above is generic and can be applied in whichever event the user is interested in
monitoring the public opinion. 

We were interested in **presidential elections** so we applied the mechanism in
US-2016 and US-2020 elections. The respective application datasets were found in:

1. https://www.kaggle.com/paulrohan2020/2016-usa-presidential-election-tweets61m-rows

2. https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets


-----------

### Project Organization
</br>

* **Descriptor directory**


-*Descriptor_getter* directory contains the following:

1. Input_dataset - A sample of cleaned tweets from the application datasets.

2. The 4 text classifiers (OLID, Political social media, Tweets with sarcasm and irony, Youtube Comments)
    used for getting the 4D tweet descriptor.

3. tweet_descriptor_getter.py - script that produces the tweet descriptors.

4. output_examples - output of tweet_descriptor_getter.py for the sample inputs in Input_dataset directory.


5. cleaning_script.ipynb - script for cleaning the original US-2016 application dataset

</br>


-The two notebooks named US_2016/2020_visualizations.ipynb were used to produce some fruitful
plots of the extracted descriptors as time series.  
  
</br>


*How to get the descriptors:*

Run on cmd : python tweet_descriptor_getter.py Input_dataset/Cleaned_dataset/2016_US_election_tweets_0_cleaned.csv

(Change the argument related to the input filename for your case if needed)

The output csv is saved in the same directory with the input csv with "_results" added to the filename


-----------------------------


* **Forecasting directory**

The same procedure/code was executed for both the US-2016 and US-2020 data. For saving space,
I included only the US-2016 experiment. Specifically you'll find:

1. code - all the necessary files to execute the experiment

2. datasets used - input data used for the experiment

3. forecasting results - results obtained from applying forecasting to US-2016 descriptors time series (testing was done on the last 7 days of the time series).  
  
  
</br>



*Steps to use this code for forecasting:*  
(All runs are performed in CMD)


Make sure the initial time_series csv has the proper structure (datasets\text_data\time_series.csv) 
and use train_test_splitt.py to split it (you need to define the train-test portions and the test portion needs 
to be equal with the forecast horizon parameter chosen for the forecasting - in my case it was 7)


**IMPORTANT**: In results\optimized_configurations directory there is a file with the optimized configs that we use for our model, so make sure you will not delete that file.   

</br>


1. In project root directory (ipath = input path, opath = output path) , (i used R-4.1.0):   
</br>

a. Run: "C:\Program Files\R\R-4.1.0\bin\Rscript.exe" preprocess_scripts/kaggle_dataset_preprocess.R

ipath=./datasets/text_data/time_series_train.csv , opath=./datasets/text_data/dataset.txt  



b. Run: "C:\Program Files\R\R-4.1.0\bin\Rscript.exe" preprocess_scripts/kaggle_results_preprocess.R

ipath=./datasets/text_data/time_series_test.csv , opath=./datasets/text_data/results.txt  




c. Run: "C:\Program Files\R\R-4.1.0\bin\Rscript.exe" preprocess_scripts/moving_window/kaggle_train_dataset_preprocess.R

ipath=./datasets/text_data/dataset.txt , opath=./datasets/text_data/moving_window/stl_7i9.txt  



d. Run: "C:\Program Files\R\R-4.1.0\bin\Rscript.exe" preprocess_scripts/moving_window/kaggle_test_dataset_preprocess.R

ipath=./datasets/text_data/dataset.txt , opath=./datasets/text_data/moving_window/test_7i9.txt  



e. Run: "C:\Program Files\R\R-4.1.0\bin\Rscript.exe" preprocess_scripts/moving_window/kaggle_validation_dataset_preprocess.R

ipath=./datasets/text_data/dataset.txt , opath=./datasets/text_data/moving_window/stl_7i9v.txt  



f. In preprocess_scripts/moving_window directory:

Run: python create_tfrecords.py , to create the 3 binary files from the 3 previously created text files

ipath=./datasets/text_data/moving_window/ , opath=./datasets/binary_data/moving_window/  
  
  
</br>

2. In project root directory:

Run: python generic_model_trainer.py --dataset_name dokimh --contain_zero_values 1 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/kaggle_web_traffic_adagrad --binary_train_file_train_mode datasets/binary_data/moving_window/stl_7i9.tfrecords --binary_valid_file_train_mode datasets/binary_data/moving_window/stl_7i9v.tfrecords --binary_train_file_test_mode datasets/binary_data/moving_window/stl_7i9v.tfrecords --binary_test_file_test_mode datasets/binary_data/moving_window/test_7i9.tfrecords --txt_test_file datasets/text_data/moving_window/test_7i9.txt --actual_results_file datasets/text_data/results.txt --input_size 9 --forecast_horizon 7 --optimizer cocob --hyperparameter_tuning smac --model_type stacking --input_format moving_window --with_accumulated_error 1 --integer_conversion 0 --address_near_zero_instability 0 --seasonality_period 7 --original_data_file datasets/text_data/dataset.txt --seed 1
(change arguments if needed)

opath= results/rnn_forecasts  


</br>

3. In utility_scripts/error_summary_scripts directory: 

Run: python ensembling_forecasts.py --dataset_name dokimh

opath= results/ensemble_rnn_forecasts  


</br>

4. In project root directory:

Run: "C:\Program Files\R\R-4.1.0\bin\Rscript.exe" error_calculator/moving_window/final_evaluation-error_calculation.R results/ensemble_rnn_forecasts/dokimh_ensemble /results/ensemble_errors/ /results/ensemble_processed_rnn_forecasts/ dokimh_ensemble_error datasets/text_data/moving_window/test_7i9.txt datasets/text_data/results.txt datasets/text_data/dataset.txt 9 7 1 0 0 7 0

opath= results/ensemble_errors  

</br>


5. In utility_scripts/error_summary_scripts directory:

Run: python error_summary_generator.py --dataset_name dokimh --is_merged_cluster_result 0

opath= results/ensemble_errors/aggregate_errors  
  
</br>



For more details regarding the execution flow and the products of it,
you can check - https://github.com/HansikaPH/time-series-forecasting


------------------------


This project was part of my diploma thesis - - and was published in the scientific journal "Social Network Analysis and Mining" - https://link.springer.com/article/10.1007/s13278-022-00922-8.




