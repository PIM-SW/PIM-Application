f = open("./GDS4602.soft")
fin = f.read().split("\n")
f.close()

healthy = fin[27].split(" = ")[1].split(",")
psoriasis = fin[32].split(" = ")[1].split(",")

healthy_tuple = []
for one, item in zip([1]*len(healthy), healthy):
    healthy_tuple.append([one, item])

psoriasis_tuple = []
for one, item in zip([-1]*len(psoriasis), psoriasis):
    psoriasis_tuple.append([one, item])

total = healthy_tuple + psoriasis_tuple
total.sort(key = lambda x : x[1])

top_array = []
for t in total:
    top_array.append(t[0])

print_array = []
print_array.append(top_array)
for line in fin[234:]:
    if(line == "!dataset_table_end"):
        break
    print_array.append(list(line.split('\t')[2:]))
import numpy as np
np_p_array = np.asarray(print_array, dtype='object')
np_p_array = np_p_array.transpose()
np_p_array.astype(float)
np.savetxt("text.out", np_p_array, fmt='%s' ,delimiter="\t")
