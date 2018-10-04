#This routine should be in the same directory as the texts
import os
import re
import sys

MAINDIR = '.' #in this directory
SubDir = [x[0] for x in os.walk('.')]
SubDir = SubDir[1:]
#check if string can be converted to float (int)
def is_float(input_):
  try:
    num = float(input_)
  except ValueError:
    return False
  return True
for sd in SubDir:
    for x in os.walk(sd):
        for f in x[2]: #Loop through the files
            newFile = open('%s/%s_new.txt'%(sd,f[:-4]), 'w') #create new files
            treat = open('%s/%s.txt'%(sd,f[:-4]), 'r') #load the original files
            for line in treat:
                if len(line.split()): #skip empty lines
                    l = re.sub(r'\([^)]*\)', '', line) #remove things within bracket
                                                       #(I am not sure about this,
                                                       #I aimed to remove things like
                                                       #(1), (b),... but it ends up
                                                       #removing things like this
                                                       #comment, which are within
                                                       #brackets)
                    l = re.sub(r'\[[^)]*\]', '', l) #remove things within bracket
                    l = l.split()
                    if len(l):
                        if is_float(l[0]): #remove first element when it's a float
                            l = l[1:]
                        newFile.write('%s\n'%(' '.join(l)))
            treat.close()
            newFile.close()
            sys.stdout.write('%s/%s.txt cleaned, saved in %s/%s_new.txt\n'%( sd,f[:-4], sd,f[:-4]))
