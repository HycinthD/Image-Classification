# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the MNIST dataset. The MNIST dataset contains handwritten digit images (0-9), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency.

## Neural Network Model

![Screenshot 2025-03-24 142009](https://github.com/user-attachments/assets/f020a62f-d478-45c6-8baa-456d3e350a51)

## DESIGN STEPS

### STEP 1:
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).
### STEP 2:
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.
### STEP 3:
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.
### STEP 4:
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.
### STEP 5:
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.
### STEP 6:
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.
### STEP 7: 
Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

### Name: HYCINTH D
### Register Number: 212223240055
```python
cclass CNNClassifier (nn.Module):
  def __init__(self):
    super (CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn. Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn. MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn. Linear (128 * 3 * 3, 128)
    self.fc2 = nn. Linear (128, 64)
    self.fc3= nn. Linear (64, 10)
  def forward(self, x):
    x = self.pool (torch.relu(self.conv1(x)))
    x = self.pool (torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size (0), -1) # Flatten the image
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```

```python
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```python
def train_model(model, train_loader, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    print('Name: HYCINTH D')
    print('Register Number: 212223240055')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch
![image](https://github.com/user-attachments/assets/4b2e89ed-2cc3-43ae-a4b4-33f7206692dd)

### Confusion Matrix

![image](https://github.com/user-attachments/assets/cf1deb87-7ed8-4104-a445-4eee4ac304bd)


### Classification Report

![image](https://github.com/user-attachments/assets/f8b7bd41-72a6-458b-a715-f3457e83fc0c)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/4703fd59-c66c-469a-a365-3eae7fd3cbf8)

## RESULT
Thus the development of a convolutional deep neural network for image classification is executed successfully.
