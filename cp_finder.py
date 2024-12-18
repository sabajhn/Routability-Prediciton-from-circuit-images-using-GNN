# Open the input file in read mode
with open("report_timing.setup.rpt", "r") as infile:
# Open the output file in write mode
    start_point = []
    end_point = []
    for line in infile:
        if line.startswith("Startpoint"):
            my_str = line.split(" ")
            print(int(my_str[1][5:-1]))
            start_point.append(int(my_str[1][5:-1]))
        if line.startswith("Endpoint"):
            my_str = line.split(" ")
            print(int(my_str[3][5:-1]))
            end_point.append(int(my_str[3][5:-1]))
            print("--------------")
    print(set(start_point),";")
    print("--------------")
    print(set(end_point),";")
