# Exercise 6-2: Spam Classification with SVMs
import numpy as np
import scipy.io as sio
from sklearn import svm

from process_email import process_email
from email_features import email_features
from get_vocablist import get_vocablist


# ==================== Part 1: Email Preprocessing ====================
print 'Preprocessing sample email (emailSample1.txt)...'

with open('emailSample1.txt') as f:
    file_contents = f.read().replace('\n', '')

word_indices = process_email(file_contents)

# Print Stats
print 'Word Indices:', word_indices


# ==================== Part 2: Feature Extraction ====================
print 'Extracting features from sample email (emailSample1.txt)...'
features = email_features(word_indices)

print 'Length of feature vector:', len(features)
print 'Number of non-zero entries:', np.sum(features > 0)


# =========== Part 3: Train Linear SVM for Spam Classification ========
# Load the Spam Email dataset
mat_data = sio.loadmat('spamTrain.mat')
X = mat_data['X']
y = mat_data['y'].ravel()

print 'Training Linear SVM (Spam Classification)...'
C = 0.1
clf = svm.LinearSVC(C=C)
clf.fit(X, y)
p = clf.predict(X)

print 'Training Accuracy:', np.mean(p == y) * 100


# =================== Part 4: Test Spam Classification ================
# Load the test dataset

mat_data = sio.loadmat('spamTest.mat')
X_test = mat_data['Xtest']
y_test = mat_data['ytest'].ravel()

print 'Evaluating the trained Linear SVM on a test set...'
p = clf.predict(X_test)

print 'Test Accuracy:', np.mean(p == y_test) * 100


# ================= Part 5: Top Predictors of Spam ====================
coef = clf.coef_.ravel()
idx = coef.argsort()[::-1]
vocab_list = get_vocablist()

print 'Top predictors of spam:'
for i in range(15):
    print "{0:<15s} ({1:f})".format(vocab_list[idx[i]], coef[idx[i]])


# =================== Part 6: Try Your Own Emails =====================
filename = 'spamSample1.txt'
with open(filename) as f:
    file_contents = f.read().replace('\n', '')

word_indices = process_email(file_contents)
x = email_features(word_indices)
p = clf.predict(x.T)
print 'Processed', filename, '\nSpam Classification:', p
print '(1 indicates spam, 0 indicates not spam)'
