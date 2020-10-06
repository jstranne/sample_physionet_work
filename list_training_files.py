import os


print("Creating List of Training Files")
directory=os.listdir('.'+os.sep+'training')
f = open('training_names.txt','w')
for folder in directory:
    if folder.startswith('tr'):
        f.write(folder+'\n')
f.close()
print("Done")
