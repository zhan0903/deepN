import subprocess

for i in range(1,2):
    cmd = "python ga.py ./configurations/ga_atari_test%s.json" % i
    returned_value = subprocess.call(cmd, shell=True)
    print('returned value:', returned_value)