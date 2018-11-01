import subprocess


cmd = "python ga.py ./configurations/ga_atari_config_debug.json"
returned_value = subprocess.call(cmd, shell=True)
print('returned value:', returned_value)
