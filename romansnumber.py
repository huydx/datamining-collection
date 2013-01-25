roman = {'I':1,'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000, 'INFI':0}

def parser(input):
  prev = 'INFI' 
  curnum = 0
  l = len(input)
  print l
  for i in range(0,l):
    curnum += roman[input[i]]
    c = input[i]
    if roman[c] > roman[prev]:
      curnum -= 2*roman[prev]
    prev = c
  return curnum

print parser("IVIX")

