import model as model
import shutil
import os, time
dir_to_watch = '/data/vanguard_FAQ/watch_dir/'
while True:
    time.sleep(1)
    files_list = os.listdir(dir_to_watch)
    print("polling...")
    for fl in files_list:
        if fl.endswith('end'):
            data_id = fl.split('.')[0]
            print("training started for id "+data_id)
            data_file = dir_to_watch+data_id+".data"
            shutil.copyfile(data_file, '../data/train_data.txt')
            shutil.copyfile(data_file, '../data/valid_data.txt')
            shutil.copyfile(data_file, '../data/train2.txt')

            model.run()
            os.remove(dir_to_watch+fl)
            os.remove(data_file)
