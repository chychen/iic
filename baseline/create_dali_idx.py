from subprocess import call
import os.path

filename_list = ['train_dataset_v2', 'validation_train_dataset_v2', 'validation_dataset']
tfrecord = "../inputs/{}.tfrecord"
tfrecord_idx = "../inputs/idx_files/{}.idx"
tfrecord2idx_script = "tfrecord2idx"

if not os.path.exists("../inputs/idx_files"):
    os.mkdir("../inputs/idx_files")

for filename in filename_list:
    if not os.path.isfile(tfrecord_idx.format(filename)):
        call([tfrecord2idx_script, tfrecord.format(filename), tfrecord_idx.format(filename)])
