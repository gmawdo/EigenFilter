import time
s = time.time()
from laspy.file import File
inFile = File("L001.las")
x = inFile.x
y = inFile.y
z = inFile.z
c = inFile.classification
e = time.time()
print(e-s)

