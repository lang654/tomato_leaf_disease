import os

base = r"C:\Users\subha\Downloads\Final Year Project"

for root, dirs, files in os.walk(base):
    print(root)