#/bin/bash
#scp -r /mnt/c/Users/jaych/Desktop/iic/* jay@10.19.104.135:/home/jay/iic
rsync -rv -e ssh --include="*/" --include="*.py" --include='*.sh' --exclude="*" /mnt/c/Users/jaych/Desktop/iic/ jay@10.19.104.135:/home/jay/iic/
