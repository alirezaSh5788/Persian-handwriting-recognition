import os

os.chdir('./dataset')
root_path = os.getcwd()

digit_folders = [str(i) for i in range(10)]
os.mkdir(os.path.join(root_path,'digit_folders'))
letter_folders = [str(i) for i in range(32)]
os.mkdir(os.path.join(root_path,'letter_folders'))

for folder in digit_folders:
    os.mkdir(os.path.join(root_path + '/digit_folders',folder))
    

for folder in letter_folders:
    os.mkdir(os.path.join(root_path+ '/letter_folders',folder))
    