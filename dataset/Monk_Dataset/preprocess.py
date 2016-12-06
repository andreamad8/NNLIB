
fileptr = open('monks2.train')
out_file = open("monks2prep.train","w")

for line in fileptr:
    i = 0
    for word in line.split():
        if(i == 0): # Output
            if(word == "1"): 
                out_file.write("1 ")
            if(word == "0"):
                out_file.write("0 ")
        if(i == 1):
            if(word == "1"):
                out_file.write("1 0 0 ")
            if(word == "2"):
                out_file.write("0 1 0 ")
            if(word == "3"):
                out_file.write("0 0 1 ")        
        elif(i == 2):
            if(word == "1"):
                out_file.write("1 0 0 ")
            if(word == "2"):
                out_file.write("0 1 0 ")
            if(word == "3"):
                out_file.write("0 0 1 ")
        elif(i == 3):
            if(word == "1"):
                out_file.write("1 0 ")
            if(word == "2"):
                out_file.write("0 1 ")
        elif(i == 4):
            if(word == "1"):
                out_file.write("1 0 0 ")
            if(word == "2"):
                out_file.write("0 1 0 ")
            if(word == "3"):
                out_file.write("0 0 1 ")
        elif(i == 5):
            if(word == "1"):
                out_file.write("1 0 0 0 ")
            if(word == "2"):
                out_file.write("0 1 0 0 ")
            if(word == "3"):
                out_file.write("0 0 1 0 ")
            if(word == "4"):
                out_file.write("0 0 0 1 ")
        elif(i == 6):
            if(word == "1"):
                out_file.write("1 0 ")
            if(word == "2"):
                out_file.write("0 1 ")
        i = i + 1
    out_file.write("\n")


out_file.close()
