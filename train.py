import numpy, scipy.io, sys, time, pandas
#from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def assign_labels(line, params):
    epoch_line = time.mktime(time.strptime(line.split(" ")[0] + " " + line.split(" ")[1], "%Y-%m-%d %H:%M:%S"))
    for i in range(0, int(params[1])):
        dates = params[i+2].split(" ")
        if time.mktime(time.strptime(dates[0],"%Y-%m-%d-%H:%M:%S")) <= epoch_line <= time.mktime(time.strptime(dates[1],"%Y-%m-%d-%H:%M:%S")):
            return dates[2]
    return 1

def readFilesCreateDataArray(params):
    list_line = []
    no_of_rows = []
    zabix_msg = [ "system.cpu.load[percpu,avg1]", "system.cpu.intr", "system.cpu.switches", "system.cpu.util[,idle]", "system.cpu.util[,interrupt]", 
        "system.cpu.util[,iowait]", "system.cpu.util[,nice]", "system.cpu.util[,softirq]", "system.cpu.util[,steal]", "system.cpu.util[,system]", 
        "system.cpu.util[,user]",  "proc.num[,,run]", "net.if.in[ens32]", "net.if.in[virbr0-nic]", "net.if.in[virbr0]", "net.if.out[ens32]",
        "net.if.out[virbr0-nic]", "net.if.out[virbr0]", "system.swap.size[,free]", "vfs.fs.inode[/,pfree]", "vfs.fs.size[/,free]", 
        "vfs.fs.size[/,used]",   "vm.memory.size[available]",  "proc.num[]",
    ]
    zabix_params =  [ 'cpu_load_start', 'cpu_intr_start', 'cpu_switches_start', 'cpu_uidle_start', 'cpu_uintr_start', 'cpu_iowait_start', 'cpu_unice_start', 
        'cpu_usoftirq', 'cpu_usteal_start', 'cpu_usystem_start', 'cpu_uuser_start', 'cpu_procrun_start', 'in_ens32_start', 'in_vibr0_nic_start', 
        'in_vibr0_start', 'out_ens32_start', 'out_vibr0_nic_start', 'out_vibr0_start', 'swap_free_start','inode_pfree_start', 'vfs_size_free_start', 
        'vfs_size_used_start', 'mem_size_start', 'cpu_proc_start'
    ]
    zabix = dict.fromkeys(zabix_params)
    data = [[] for i in range(24)]

    #find the rows containing features one by one
    with open(params[0]) as f:
        list_line = (f.read()).splitlines()
        for line in list_line:
            for i in range(0, len(zabix_msg)):
                if "Found item " + zabix_msg[i] + " on host" in line:
                    zabix[zabix_params[i]] = list_line.index(line) + 2
                    no_of_rows.append(int(list_line[zabix[zabix_params[i]]].split(" ")[1]))
        #create list containing time, feature value1, feature value2....lists.
        for j in range(1, min(no_of_rows)+1):
            for i in range(0, 23):
                data[i].append(float(list_line[j + zabix[zabix_params[i]]].split(" ")[3]))
            data[23].append(int(assign_labels(list_line[j + zabix[zabix_params[i]]], params))) #label of the training element
    return data

#traspose
data = []
X = numpy.empty((0, 24), float)
with open("data/argfile") as f:
    aline = f.readline()
    while aline != "":
        params = []
        params.append('data/vmstats/' + aline.rstrip("\n"))
        num  = int(f.readline())
        params.append(num)
        for i in range(0, num):
            params.append(f.readline().rstrip("\n"))
        data = readFilesCreateDataArray(params)
        X = numpy.concatenate((X, numpy.array(data).transpose()), 0)
        aline = f.readline()

names = [ 'cpu_load', 'cpu_interrupts', 'cpu_switches', 'cpu_uidle', 'cpu_uintr', 'cpu_uiowait', 'cpu_unice', 'cpu_usoftirq', 'cpu_usteal', 'cpu_usystem', 'cpu_uuser', 
    'cpu_procrun', 'in_ens32', 'in_vibr0_nic', 'in_vibr0', 'out_ens32', 'out_vibr0_nic', 'swap_free', 'inode_pfree', ' vfs_size_free', 'vfs_size_used', 'mem_size', 'class'
]
dataset = pandas.DataFrame(X[:,1:], columns = names, index=X[:,0])

# Split-out validation dataset
X = dataset.values[:,0:len(X[0])-1]
Y = dataset.values[:,len(X[0])-1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
scoring = 'accuracy'

# Spot Check Algorithms
models = { 'LR': LogisticRegression(), 'LDA': LinearDiscriminantAnalysis(), 'KNN': KNeighborsClassifier(), 'DTC': DecisionTreeClassifier(), 
           'NB': GaussianNB(), 'SVM': SVC(), 'MLP': MLPClassifier() }

# evaluate each model in turn
results = []
names = []
for name in models.keys():
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(models[name], X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Make predictions on validation dataset
for name in models.keys():
    print(name)
    func = models[name]
    func.fit(X_train, Y_train)
    predictions = func.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
