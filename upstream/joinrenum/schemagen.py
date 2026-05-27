from random import randint


relation_num = 4
variable_num = 8
def gen_schema(relation_num, variable_num,factor = 3):
    with open("db/relations.txt", "w") as f:
        for i in range(relation_num):
            f.write("R" + str(i+1) + "(")
            # sample variables for each relation
            variables = []
            for j in range(variable_num):
                if randint(1,factor) == 1: variables.append("V" + str(j+1))
            f.write(",".join(variables))
            f.write(")\n")
    with open("db/numlines.txt", "w") as f:
        for i in range(relation_num):
            f.write("R" + str(i+1) + " 0" + "\n")
    with open("db/filenames.txt", "w") as f:
        for i in range(relation_num):
            f.write("R" + str(i+1) + " db/R" + str(i+1) + ".tbl" + "\n")

gen_schema(relation_num, variable_num)