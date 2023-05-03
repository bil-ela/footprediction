import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


df = pd.read_csv('dataset_V2.csv')
df = df.dropna(inplace=False)

groupes = df.groupby('FTR')
taille_echantillon = min([len(groupe) for _, groupe in groupes])
df = pd.concat([groupe.sample(taille_echantillon) for _, groupe in groupes])

df_equipe = df[['HomeTeam', 'AwayTeam', 'FTR', 'WHH', 'WHD', 'WHA']]
df = df.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTR','HomeFormPtsStr','AwayFormPtsStr','MatchWeek'], axis=1)

X = df
y = df_equipe['FTR']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
encoded_dict = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

X = X.astype(np.float32)
y = y.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_np = X.to_numpy()
y_np = y

model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(41,), kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
cvscores = []
for train, test in kfold.split(X_np, y_np):
    model.fit(X_np[train], y_np[train], epochs=20, batch_size=64, verbose = 2, validation_data=(X_np[test], y_np[test]))
    scores = model.evaluate(X_np[test], y_np[test], verbose=0)
    cvscores.append(scores[1])

test_acc_RN_cross = cvscores.mean()
print(test_acc_RN_cross)