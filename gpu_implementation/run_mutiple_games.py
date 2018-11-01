import subprocess


cmd0 = "python ga.py ./configurations/ga_atari_config_debug.json"
cmd1 = "python ga.py ./configurations/ga_atari_debug.json"

for i in range(2):
    cmd = cmd+"%d"%i
    print(cmd)
    returned_value = subprocess.call(cmd, shell=True)
    print('returned value:', returned_value)
