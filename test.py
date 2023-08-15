from auto_mi.tasks import IntegerGroupFunctionRecoveryTask

t = IntegerGroupFunctionRecoveryTask(1023, 5)
e = t.get_dataset(0)

print(e[0])