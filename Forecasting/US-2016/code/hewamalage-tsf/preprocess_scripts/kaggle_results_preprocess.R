OUTPUT_DIR="./datasets/text_data/"

output_file_name = 'results.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

file <-read.csv(file="./datasets/text_data/time_series_test.csv",sep=',',header = TRUE)
result_dataset <-as.data.frame(file[,-1])

result_dataset[is.na(result_dataset)] = 0

print(head(result_dataset))

# printing the results to the file
write.table(result_dataset, output_file_full_name, sep = ";", row.names = TRUE, col.names = FALSE)
