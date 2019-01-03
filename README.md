GDG Machine learning of Time-serise data study group repository

Tutorial

Ref : https://nolboo.kim/blog/2013/10/06/github-for-beginner/


0. Configure user setting

>$ git config --global user.name "Your Name Here"
>$ git config --global user.email "your_email@youremail.com"

1. Move to a target folder and start a local git repository

>$ git init


2. Link to the online repository

>$ git remote add origin https://github.com/schwarzg/gdg_sigongmo.git


3. Check the access to the online repository

>$ git remote -v


4. Make a brance that you manage

>$ git branch yourid


5. Change the status into your branch

>$ git checkout yourid


6. Make documents and files

>filename.txt


7. Add file to repository

>$ git add filename.txt
>$ git add . ( if you want to add all files in the folder )


8. Commit (update the change in file)

>$ git commit -m "Write what is changing"


9. Upload your files to online repository

>$ git push origin yourid

