import os
input_filename_prefix = "input"
target_filename_prefix = "target"
suffix = ["00", "01", "02", "03", "04",
          "05", "06", "07", "08", "09", "10", "11"]
txt = ".txt"
methods = ["A_h1", "A_h2", "IDA_h1", "IDA_h2"]
count = 0
for method in methods:
    print("now executing: ", end='')
    print(method)
    for suf in suffix:
        input_filename = input_filename_prefix + suf + txt
        target_filename = target_filename_prefix + suf + txt
        os.system("./a.out " + method + " " + input_filename + " " + target_filename)
