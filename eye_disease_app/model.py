import torch
import torch.nn as nn

class StudentCNN(nn.Module):
        def __init__(self, num_classes=6):
            super(StudentCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(48)
            self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 14 * 14, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))
            x = self.pool(torch.relu(self.bn4(self.conv4(x))))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x