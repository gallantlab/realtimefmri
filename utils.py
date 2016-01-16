import os

package_directory = '/Users/robert/Documents/gallant/realtimefmri'

def get_database_directory():
	return os.path.join(package_directory, 'database')

def get_subject_directory(subject):
	return os.path.join(get_database_directory(), subject)

def get_example_data_directory():
	return os.path.join(package_directory, 'benchmark_data', 'pixeldata_07')

def get_log_directory():
	return os.path.join(package_directory, 'log')

def get_test_data_directory():
	return os.path.join(package_directory, 'tests/data')

def generate_command(command, params):
	cmd = [command]

	kw_args = [i for i in params if ('flag' in i) and ('value' in i)]
	first_pos_args = [i for i in params if i.get('position', None)=='first']
	last_pos_args = [i for i in params if i.get('position', None)=='last']
	flag_args = [i for i in params if not 'value' in i]

	for a in first_pos_args:
		cmd.append(a['value'])

	for a in kw_args:
		cmd.extend(['-'+a['flag'], a['value']])

	for a in flag_args:
		cmd.append('-'+a['flag'])

	for a in last_pos_args:
		cmd.append(a['value'])

	return cmd
