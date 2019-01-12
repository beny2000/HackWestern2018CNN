data = open('data.txt')
data_set =[]
for line in data:
    line = line.strip().split('/')
    data_set.append([line[2],line[6]])

print(data_set)


