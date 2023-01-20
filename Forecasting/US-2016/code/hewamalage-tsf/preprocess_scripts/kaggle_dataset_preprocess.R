OUTPUT_DIR="./datasets/text_data/"

file <-read.csv(file="./datasets/text_data/time_series_train.csv",sep=',',header = TRUE)
dataset <-as.data.frame(file[,-1])

output_file_name = 'dataset.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

dataset[is.na(dataset)] = 0

# printing the dataset to the file
write.table(dataset, output_file_full_name, sep = ",", row.names = FALSE, col.names = FALSE)
