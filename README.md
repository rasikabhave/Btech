# Project Directory Structure and Description

### trained_model_testing.py
	_reads data from foo.csv_
	_trains models on that data_
	_gets zabbix data / minute_
	_selects features from that data_
	_predicts classes of that data_
	_sends email if it is an anomaly_

### train.py
	_Reads each file written in argfile_
	_Creates datapoints and selects features and labels those datapoints._
	_Trains models on that data and prints testing data_

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
