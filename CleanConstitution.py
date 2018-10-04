#This routine should be in the same directory as the texts
import re
import sys

#check if string can be converted to float (int)
def is_float(input_):
  try:
    num = float(input_)
  except ValueError:
    return False
  return True


Languages = ['Afrikaans', 'English', 'isiNdebele',  'isiXhosa',
             'isiZulu', 'Sepedi', 'Sesotho', 'Setswana', 'Siswati',
             'Tshivenda', 'Xitsonga']

for f in Languages: #Loop through the files
    newFile = open('%s_new.txt'%f, 'w') #create new files
    treat = open(f+'.txt', 'r') #load the original files
    for line in treat:
        if len(line.split()): #skip empty lines
            l = re.sub(r'\([^)]*\)', '', line) #remove things within bracket
                                               #(I am not sure about this,
                                               #I aimed to remove things like
                                               #(1), (b),... but it ends up
                                               #removing things like this
                                               #comment, which are within
                                               #brackets)
            l = l.split()
            if len(l):
                if is_float(l[0]): #remove first element when it's a float
                    l = l[1:]
                newFile.write('%s\n'%(' '.join(l)))
    treat.close()
    newFile.close()
    sys.stdout.write('%s cleaned, saved in %s\n'%(f+'.txt', f+'_new.txt'))
