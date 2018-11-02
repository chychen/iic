#!/bin/bash
result_file_path="jay_result.csv"
python eval.py \
--restore_path="log/v2/model.ckpt-146080" \
--result_path=$result_file_path \
--test_data_folder_path="../inputs/stage_1_test_images"
# kaggle competitions submit -c inclusive-images-challenge -f $result_file_path -m "146080"
