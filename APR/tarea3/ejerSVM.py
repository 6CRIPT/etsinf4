from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

C_values = [0.1, 1, 10]
kernel_types = ['linear', 'rbf']
results = {}


for C in C_values:
    for kernel in kernel_types:
        clf = svm.SVC(C=C, kernel=kernel).fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)
        y_test_pred = clf.predict(x_test)
        err_train = 1 - accuracy_score(y_train, y_train_pred)
        err_test = 1 - accuracy_score(y_test, y_test_pred)
        results[(C, kernel)] = {'err_train': err_train, 'err_test': err_test}

fig, ax = plt.subplots(len(C_values), len(kernel_types), figsize=(10, 8))

for i, C in enumerate(C_values):
    for j, kernel in enumerate(kernel_types):
        result = results[(C, kernel)]
        ax[i, j].bar(['Train', 'Test'], [result['err_train'], result['err_test']], color=['blue', 'orange'])
        ax[i, j].set_title(f'C={C}, Kernel={kernel}')
plt.tight_layout()
plt.show()
