#bin/sh$
#set -e$

#git init
#git remote add origin git@git.tafsm.org:progress/jwang-progress.git

text=$1

git add .
git commit -m "$text"
git push -u origin main
