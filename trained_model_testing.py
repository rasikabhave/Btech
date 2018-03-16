#loop once for every minute
	#done: import the get metrics cha code...but limit it to only one entry
	#pass that entry to the model and print if it is an anomaly or not
	

#! /usr/bin/env python3.4
import numpy
import scipy.io
import sys
import time
import pandas
import threading
import signal
import sched
import logging
import base64
import smtplib
import threading
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
from zabbix.api import ZabbixAPI

def send_mail(anomaly_num):
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.ehlo()
	server.starttls()
	#Next, log in to the server
	server.login("nikifun24", base64.b64decode("Q2hhbXBpb24yNA=="))
	#print anomaly_num
	#Send the mail
	msg = "\n" + str(anomaly_num) + ": anomaly took place right now!" # The /n separates the message from the headers
	server.sendmail("nikifun24@gmail.com", "nikitaparanjape@gmail.com", msg)
	server.quit()
	return

def get_classify_one_datapoint(trained_models):
#getting the data
	#create varialbles to authenticate later
	url = "http://10.1.8.88/zabbix"
	user = "admin"
	password = "zabbix"
	host = "Server2"
	key = "vfs.fs.size[/,free]"
	use_older_authenticate_method = False

	# Create ZabbixAPI class instance
	zapi = ZabbixAPI(url, use_older_authenticate_method, user, password)
	thehost = zapi.do_request('host.get',
		                      {
		                          'filter': {'host': host},
		                          'selectItems' : 'extend',
		                          'output': 'extend'
		                      })
	if len(thehost['result'])<1:
	  print "HALTING. There was no host defined in zabbix with id: {}".format(host)
	  sys.exit(2)
	hostId = thehost['result'][0]['hostid']

	#get metrics for that host
	history = zapi.do_request('item.get',
		                      {
		    "output": "extend",
		    "hostids": hostId,
		    "sortfield": 'name'
	})
	print len(history['result'])
	#following are the metrics that we need to pass to our algo
	items_we_need = ['system.cpu.load[percpu,avg1]','system.cpu.intr','system.cpu.switches','system.cpu.util[,idle]','system.cpu.util[,interrupt]','system.cpu.util[,iowait]','system.cpu.util[,nice]','system.cpu.util[,softirq]','system.cpu.util[,steal]','system.cpu.util[,system]','system.cpu.util[,user]','proc.num[,,run]','net.if.in[ens32]','net.if.in[virbr0-nic]','net.if.in[virbr0]','net.if.out[ens32]','net.if.out[virbr0-nic]','net.if.out[virbr0]','system.swap.size[,free]','vfs.fs.inode[/,pfree]','vfs.fs.size[/,free]','vfs.fs.size[/,used]','vm.memory.size[available]']

	#create a list of the metrics we need from the fetched list
	list1 = []
	for i in range(0, len(history['result'])):
		for item in items_we_need:
			if history['result'][i]['key_'] == item:
				list1.append(float(history['result'][i]['lastvalue']))
				print history['result'][i]['lastvalue'],history['result'][i]['key_'], history['result'][i]['lastclock']

	#return this value			
	X_test = numpy.array(list1)
	
#classifying the data:	
	for model in trained_models:
		predicted_class = int(model.predict(X_test.reshape(1, -1)))
		print "predicted_class = " , predicted_class
		
		#if predicted_class != 1:
			#t = threading.Thread(target = send_mail, args = (predicted_class, ))
			#t.start()	
	print " \n"	
class PeriodicEvent(object):
        def __init__(self, interval, func, parameter_list):
                self.interval = interval
                self.func = func
                self.terminate = threading.Event()
                self.parameter_list = parameter_list
        def _signals_install(self, func):
                for sig in [signal.SIGINT, signal.SIGTERM]:
                        signal.signal(sig, func)

        def _signal_handler(self, signum, frame):
                self.terminate.set()

        def run(self):
                self._signals_install(self._signal_handler)
                while not self.terminate.is_set():
                        self.func(self.parameter_list)
                        self.terminate.wait(self.interval)
                self._signals_install(signal.SIG_DFL)
                
def create_a_model_of_the_data():
	#open file where all the data is stored
	url  = "/home/nachiket/foo.csv"
	names = ['cpu_load', 'cpu_interrupts', 'cpu_switches', 'cpu_uidle', 'cpu_uintr', 'cpu_uiowait', 'cpu_unice', 'cpu_usoftirq', 'cpu_usteal', 'cpu_usystem', 'cpu_uuser', 'cpu_procrun', 'in_ens32', 'in_vibr0_nic', 'in_vibr0', 'out_ens32', 'out_vibr0_nic','out_vibr0' 'swap_free', 'inode_pfree', ' vfs_size_free', 'vfs_size_used', 'mem_size', 'class']
	dataset = pandas.read_csv(url, index_col=None)
	array = dataset.values
	#separate the input and expected classes
	X = array[:,0:len(array[0])-1]
	Y = array[:,len(array[0])-1]
	# Test options and evaluation metric
	seed = 7
	scoring = 'accuracy'
	#fit the model to the data
	lr = LogisticRegression().fit(X,Y)
	lda = LinearDiscriminantAnalysis().fit(X,Y)
	knn = KNeighborsClassifier().fit(X,Y)
	dtc = DecisionTreeClassifier().fit(X,Y)
	nb = GaussianNB().fit(X,Y)
	svm = SVC().fit(X,Y)
	trained_models = [lr, lda, knn, dtc, nb, svm]
	return trained_models
	
def main():
        trained_models = create_a_model_of_the_data() 
        task = PeriodicEvent(60, get_classify_one_datapoint, trained_models)
        task.run()
        return 0

if __name__ == "__main__":
        exit(main())
