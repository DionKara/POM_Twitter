from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter
import os

output_path = "../../datasets/binary_data/moving_window/"

    
if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        input_size = 9,
        output_size = 7,
        train_file_path = '../../datasets/text_data/moving_window/stl_7i9.txt',
        validate_file_path = '../../datasets/text_data/moving_window/stl_7i9v.txt',
        test_file_path = '../../datasets/text_data/moving_window/test_7i9.txt',
        binary_train_file_path = output_path + 'stl_7i9.tfrecords',
        binary_validation_file_path = output_path + 'stl_7i9v.tfrecords',
        binary_test_file_path = output_path + 'test_7i9.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()


