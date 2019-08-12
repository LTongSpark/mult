#-*-encoding:utf-8-*-

list1 = [{'flag': 0.01, 'num': 0.4}, {'flag': 0.1, 'num': 0.6}, {'flag': 1, 'num': 0.7}, {'flag': 10, 'num': 0.8}, {'flag': 100, 'num': 0.9}]

print(min(list1 ,key=lambda x :x['num']))
print()
