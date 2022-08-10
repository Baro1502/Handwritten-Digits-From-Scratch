import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''
w: weight
b: bias
i: input
o: output
h: hidden (layer)
'''

df = pd.read_csv('dataset.csv')
csv = df.to_numpy()
np.random.shuffle(csv)

images = csv[:,0:-1]
labels = csv[:,0]
a = np.arange(10)
list_of_onehots = []
for label in labels:
    list_of_onehots.append((a==label).astype(np.int32))

one_hot = np.array(list_of_onehots)


w_h_i = np.random.uniform(-0.5, 0.5, (100, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 100))
b_h_i = np.zeros((100, 1))
b_h_o = np.zeros((10, 1))

learn_rate = 0.01
correct_point = 0
epochs = 3
for epoch in range(epochs):
    for X_TRAIN, Y_TRAIN in zip(images, one_hot):
        # Increase dimension to become matrix-like
        X_TRAIN.shape += (1,)
        Y_TRAIN.shape += (1,)
        
        # Forward propagation input -> hidden
        h_pre = b_h_i + (w_h_i @ X_TRAIN)
        h_act = 1 / (1 + np.exp(-h_pre))

        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h_act
        o_act = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        e = np.sum((o_act - Y_TRAIN) ** 2, axis=0) / len(o_act)
        correct_point += int(np.argmax(o_act) == np.argmax(Y_TRAIN))
        
        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o_act - Y_TRAIN
        w_h_o += -learn_rate * (delta_o @ np.transpose(h_act))
        b_h_o += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h_act * (1 - h_act))
        w_h_i += -learn_rate * delta_h @ np.transpose(X_TRAIN)
        b_h_i += -learn_rate * delta_h

    # Show accuracy for this epoch
    acc = round((correct_point / images.shape[0]) * 100, 2)
    print(f"Acc: {acc}%")
    correct_point = 0

# Show results
while True:
    index = int(input(f"Enter a number (0 - {labels.shape[0]-1})(-1 to exit): "))
    if index == -1:
        exit()
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_h_i + w_h_i @ img.reshape(784, 1)
    h_act = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h_act
    o_act = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Subscribe if its a {o_act.argmax()} :)")
    plt.show()