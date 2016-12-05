import urllib.request
import sys
import tarfile
import os

file_name = "kifu.gam.tar.gz"
train_dir_name = "kifu"

def download(file_name):

    url = "http://starlancer.org/~is2004/owiki/?page=%A3%B5%A3%B0%CB%FC%B4%FD%C9%E8%B7%D7%B2%E8&file=kifu%2Egam%2Etar%2Egz&action=ATTACH"
    urllib.request.urlretrieve(url,file_name)

def extract(file_name,train_dir_name):
    if (file_name.endswith("tar.gz")):
        if(not os.path.exists(train_dir_name)):
            os.mkdir(train_dir_name)

        tar = tarfile.open(file_name, "r:gz")
        tar.extractall(train_dir_name+"/")
        tar.close()

download(file_name)
extract(file_name,train_dir_name)
