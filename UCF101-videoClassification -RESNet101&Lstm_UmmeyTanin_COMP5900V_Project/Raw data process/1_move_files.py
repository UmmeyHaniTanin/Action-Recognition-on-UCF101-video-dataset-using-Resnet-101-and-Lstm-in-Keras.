"""
SelfNotes :  run once to move all the files into
the appropriate train/test folders.

"""
import os
import os.path

def get_train_test_lists(version='01'):
    
    test_file = os.path.join('ucfTrainTestlist', 'testlist' + version + '.txt')
    train_file = os.path.join('ucfTrainTestlist', 'trainlist' + version + '.txt')

    
    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)]

    
    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list]

    file_groups = {
        'train': train_list,
        'test': test_list
    }

    return file_groups

def move_files(file_groups):
    
    # Do each of our groups.
    for group, videos in file_groups.items():

        # Do each of our videos.
        for video in videos:

            parts = video.split(os.path.sep)
            classname = parts[0]
            filename = parts[1]

            # Check if this class exists.
            if not os.path.exists(os.path.join(group, classname)):
                print("Creating folder for %s/%s" % (group, classname))
                os.makedirs(os.path.join(group, classname))

            if not os.path.exists(filename):
                print("Can't find %s to move. Skipping." % (filename))
                continue

            dest = os.path.join(group, classname, filename)
            print("Moving %s to %s" % (filename, dest))
            os.rename(filename, dest)

    print("Done.")

def main():
    
    
    group_lists = get_train_test_lists()

    # Moving files.
    move_files(group_lists)

if __name__ == '__main__':
    main()
