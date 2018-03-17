# Project Directory Structure and Description

### trained_model_testing.py
_Reads data from foo.csv trains models on that data gets zabbix data / minute. Selects features from that data predicts classes of that data sends email if it is an anomaly_

### train.py
_Reads each file written in argfile. Creates datapoints and selects features and labels those datapoints. Trains models on that data and prints testing data_

### data/argfile
_This is used by train.py to label the datapoints._
_It has the following structure it can have any number of such structs one after another_
		
		{	
			data_file_name
			num_of_lines_following_this_line
			time1 time2 anomaly_code
			time1 time2 anomaly_code
			.
			.	
			.	
		}

### data/vmstats/
_Stats and metrics generted by zabix of the VM_
