import numpy
import scipy.io
import sys
import time
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
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
from sklearn.neural_network import MLPClassifier


def assign_labels(line, params):
	date_in_line = line.split(" ")[0] + " " + line.split(" ")[1]
	epoch_line = time.mktime(time.strptime(date_in_line, "%Y-%m-%d %H:%M:%S"))
	for i in range(0, int(params[1])):
		dates = params[i+2].split(" ")
		epoch1 = time.mktime(time.strptime(dates[0],"%Y-%m-%d-%H:%M:%S"))
		epoch2 = time.mktime(time.strptime(dates[1],"%Y-%m-%d-%H:%M:%S"))
		if epoch_line >= epoch1 and epoch_line <=epoch2:
			return dates[2]
	return 1

def readFilesCreateDataArray(params):
	#declare variables
	list_line = []
	no_of_rows = []
	#file input
	f = open(params[0])
	pend = f.read()
	list_line = pend.splitlines()

	#find the rows containing features one by one
	for i in range (1, len(list_line)):
		if "Found item system.cpu.load[percpu,avg1] on host" in list_line[i]:
			cpu_load_start = i+2
			no_of_rows.append(int(list_line[cpu_load_start].split(" ")[1]))
		elif "Found item system.cpu.intr on host" in list_line[i]:
			cpu_intr_start = i+2
			no_of_rows.append(int(list_line[cpu_intr_start].split(" ")[1]))
		elif "Found item system.cpu.switches on host" in list_line[i]:
			cpu_switches_start = i+2
			no_of_rows.append(int(list_line[cpu_switches_start].split(" ")[1]))
		elif "Found item system.cpu.util[,idle] on host" in list_line[i]:
			cpu_uidle_start = i+2
			no_of_rows.append(int(list_line[cpu_uidle_start].split(" ")[1]))
		elif "Found item system.cpu.util[,interrupt] on host"  in list_line[i]:
			cpu_uintr_start = i+2
			no_of_rows.append(int(list_line[cpu_uintr_start].split(" ")[1]))
		elif "Found item system.cpu.util[,iowait] on host" in list_line[i]:
			cpu_uiowait_start = i+2
			no_of_rows.append(int(list_line[cpu_uiowait_start].split(" ")[1]))
		elif "Found item system.cpu.util[,nice] on host" in list_line[i]:
			cpu_unice_start = i+2
			no_of_rows.append(int(list_line[cpu_unice_start].split(" ")[1]))
		elif "Found item system.cpu.util[,softirq] on host" in list_line[i]:
			cpu_usoftirq_start = i+2
			no_of_rows.append(int(list_line[cpu_usoftirq_start].split(" ")[1]))
		elif "Found item system.cpu.util[,steal] on host" in list_line[i]:
			cpu_usteal_start = i+2
			no_of_rows.append(int(list_line[cpu_usteal_start].split(" ")[1]))
		elif "Found item system.cpu.util[,system] on host" in list_line[i]:
			cpu_usystem_start = i+2
			no_of_rows.append(int(list_line[cpu_usystem_start].split(" ")[1]))
		elif "Found item system.cpu.util[,user] on host" in list_line[i]:
			cpu_uuser_start = i+2
			no_of_rows.append(int(list_line[cpu_uuser_start].split(" ")[1]))
		elif "Found item proc.num[,,run] on host" in list_line[i]:
			cpu_procrun_start = i+2
			no_of_rows.append(int(list_line[cpu_procrun_start].split(" ")[1]))
		elif "Found item proc.num[] on host" in list_line[i]:
			cpu_proc_start = i+2
			no_of_rows.append(int(list_line[cpu_proc_start].split(" ")[1]))
		elif "Found item net.if.in[ens32] on host" in list_line[i]:
			in_ens32_start = i+2
			no_of_rows.append(int(list_line[in_ens32_start].split(" ")[1]))
		elif "Found item net.if.in[virbr0-nic] on host" in list_line[i]:
			in_vibr0_nic_start = i+2
			no_of_rows.append(int(list_line[in_vibr0_nic_start].split(" ")[1]))
		elif "Found item net.if.in[virbr0] on host" in list_line[i]:
			in_vibr0_start = i+2
			no_of_rows.append(int(list_line[in_vibr0_start].split(" ")[1]))
		elif "Found item net.if.out[ens32] on host" in list_line[i]:
			out_ens32_start = i+2
			no_of_rows.append(int(list_line[out_ens32_start].split(" ")[1]))
		elif "Found item net.if.out[virbr0-nic] on host" in list_line[i]:
			out_vibr0_nic_start = i+2
			no_of_rows.append(int(list_line[out_vibr0_nic_start].split(" ")[1]))
		elif "Found item net.if.out[virbr0] on host" in list_line[i]:
			out_vibr0_start = i+2
			no_of_rows.append(int(list_line[out_vibr0_start].split(" ")[1]))
		elif "Found item system.swap.size[,free] on host" in list_line[i]:
			swap_free_start = i+2
			no_of_rows.append(int(list_line[swap_free_start].split(" ")[1]))
		elif "Found item vfs.fs.inode[/,pfree] on host" in list_line[i]:
			inode_pfree_start = i+2
			no_of_rows.append(int(list_line[inode_pfree_start].split(" ")[1]))
		elif "Found item vfs.fs.size[/,free] on host" in list_line[i]:
			vfs_size_free_start = i+2
			no_of_rows.append(int(list_line[vfs_size_free_start].split(" ")[1]))
		elif "Found item vfs.fs.size[/,used] on host" in list_line[i]:
			vfs_size_used_start = i+2
			no_of_rows.append(int(list_line[vfs_size_used_start].split(" ")[1]))
		elif "Found item vm.memory.size[available] on host" in list_line[i]:
			mem_size_start = i+2
			no_of_rows.append(int(list_line[mem_size_start].split(" ")[1]))
	#find out the amount of data present
	min_num_of_rows = min(no_of_rows)
	#create list containing time, feature value1, feature value2....lists.
	data = []
	for j in range(0, 24):
		data.append([])
	for j in range(1, min_num_of_rows+1):
		data[0].append(float(list_line[j + cpu_load_start].split(" ")[3]))
		data[1].append(float(list_line[j + cpu_intr_start].split(" ")[3]))
		data[2].append(float(list_line[j + cpu_switches_start].split(" ")[3]))
		data[3].append(float(list_line[j + cpu_uidle_start].split(" ")[3]))
		data[4].append(float(list_line[j + cpu_uintr_start].split(" ")[3]))
		data[5].append(float(list_line[j + cpu_uiowait_start].split(" ")[3]))
		data[6].append(float(list_line[j + cpu_unice_start].split(" ")[3]))
		data[7].append(float(list_line[j + cpu_usoftirq_start].split(" ")[3]))
		data[8].append(float(list_line[j + cpu_usteal_start].split(" ")[3]))
		data[9].append(float(list_line[j + cpu_usystem_start].split(" ")[3]))
		data[10].append(float(list_line[j + cpu_uuser_start].split(" ")[3]))
		data[11].append(float(list_line[j + cpu_procrun_start].split(" ")[3]))
		data[12].append(float(list_line[j + in_ens32_start].split(" ")[3]))
		data[13].append(float(list_line[j + in_vibr0_nic_start].split(" ")[3]))
		data[14].append(float(list_line[j + in_vibr0_start].split(" ")[3]))
		data[15].append(float(list_line[j + out_ens32_start].split(" ")[3]))
		data[16].append(float(list_line[j + out_vibr0_nic_start].split(" ")[3]))
		data[17].append(float(list_line[j + out_vibr0_start].split(" ")[3]))
		data[18].append(float(list_line[j + swap_free_start].split(" ")[3]))
		data[19].append(float(list_line[j + inode_pfree_start].split(" ")[3]))
		data[20].append(float(list_line[j + vfs_size_free_start].split(" ")[3]))
		data[21].append(float(list_line[j + vfs_size_used_start].split(" ")[3]))
		data[22].append(float(list_line[j + mem_size_start].split(" ")[3]))
		data[23].append(int(assign_labels(list_line[j + cpu_proc_start], params)))
	f.close()
	return data
		
#traspose
data = []
X = numpy.empty((0, 24), float)
f = open("argfile")
aline = f.readline()
while aline != "":
	params = []
	params.append(aline.rstrip("\n"))
	num  = int(f.readline())
	params.append(num)
	for i in range(0, num):
		params.append(f.readline().rstrip("\n"))
	data = readFilesCreateDataArray(params)
	X = numpy.concatenate((X, numpy.array(data).transpose()), 0)
	aline = f.readline()
f.close()
#write it to a .csv file
numpy.savetxt("foo.csv", X, delimiter = ",")
scipy.io.savemat('/home/nachiket/data5.mat', mdict={'X':X})
# Load dataset
url  = "/home/nachiket/foo.csv";
names_li = ['cpu_load', 'cpu_interrupts', 'cpu_switches', 'cpu_uidle', 'cpu_uintr', 'cpu_uiowait', 'cpu_unice', 'cpu_usoftirq', 'cpu_usteal', 'cpu_usystem', 'cpu_uuser', 'cpu_procrun', 'in_ens32', 'in_vibr0_nic', 'in_vibr0', 'out_ens32', 'out_vibr0_nic','out_vibr0','swap_free', 'inode_pfree', ' vfs_size_free', 'vfs_size_used', 'mem_size','class']
#dataset = pandas.read_csv(url, names=names, index_cols = False)
print len(names_li)
dataset = pandas.read_csv(url, names = names_li, index_col=None,header=None)
print(dataset.shape)
# shape
#print(dataset.shape)

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('class').size())

# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
#plt.show()

# histograms
#dataset.hist()
#plt.show()

# scatter plot matrix
#scatter_matrix(dataset)
#plt.show()


# Split-out validation dataset
array = dataset.values
X = array[:,0:len(X[0])-1]
Y = array[:,len(X[0])]
print X[0]
print Y[0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('NN', MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(25,), random_state=1)))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
print "evaluated teh models"	
# Compare Algorithms

#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()

# Make predictions on validation dataset
print "NN:"
nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(25,), random_state=1)
nn.fit(X_train, Y_train)
predictions = nn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print "LR:"
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print "LDA:"
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print "KNN:"
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print "DTC:"
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
predictions = dtc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print "NB:"
nb = GaussianNB()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


print "SVM:"
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

