import torch
import torch.nn as nn
from transformers.models.deit import DeiTModel

DEIT_MODEL_CHECKPOINT = "facebook/deit-tiny-distilled-patch16-224"


class PPOBaseModel(nn.Module):
    def __init__(self):
        super(PPOBaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit: DeiTModel = DeiTModel.from_pretrained(DEIT_MODEL_CHECKPOINT)  # type: ignore
        self.vit.to(self.device)  # type: ignore
        self.vit.train()  # type: ignore

    def forward(self, imgs: torch.Tensor):  # type: ignore
        last_hidden_state = self.vit(imgs).last_hidden_state
        return last_hidden_state[:, 0]


class SimpleCNN(nn.Module):
    def __init__(self, action_size: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 56 * 56, 128)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(128, action_size)

    def forward(self, imgs: torch.Tensor):
        x = self.conv1(imgs)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu3(x)
        x = self.out(x)
        x = nn.functional.softmax(x, dim=-1)
        return x


class SSAgentModel(nn.Module):
    def __init__(self, embedding_size: int = 512, action_size: int = 7):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.start_state_encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 112x112
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 56x56
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 28x28
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 14x14
        #     nn.ReLU(True),
        #     nn.Conv2d(
        #         128, embedding_size, kernel_size=7, stride=1, padding=0
        #     ),  # 8x8 (output size can be adjusted by kernel size)
        #     nn.Flatten(),
        #     nn.Linear(8 * 8 * embedding_size, embedding_size),
        # )
        # self.goal_state_encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 112x112
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 56x56
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 28x28
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 14x14
        #     nn.ReLU(True),
        #     nn.Conv2d(
        #         128, embedding_size, kernel_size=7, stride=1, padding=0
        #     ),  # 8x8 (output size can be adjusted by kernel size)
        #     nn.Flatten(),
        #     nn.Linear(8 * 8 * embedding_size, embedding_size),
        # )
        # self.action_predictor = nn.Sequential(
        #     nn.Linear(2 * embedding_size, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, action_size),
        #     nn.Softmax(dim=-1),
        # )
        self.simp = SimpleCNN(action_size=action_size)
        # self.img_encoder = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 112 x 112
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     # nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 56 x 56
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 28 x 28
        #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        #     # nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 14 x 14
        #     # nn.Dropout(p=0.5),
        #     nn.Flatten(),
        #     nn.Linear(512 * 14 * 14, 512),
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(2 * 512, 512),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(p=0.5),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(p=0.5),
        #     nn.Linear(256, action_size),
        #     nn.Softmax(dim=-1),
        # )

    def forward(self, start_imgs: torch.Tensor, goal_imgs: torch.Tensor):
        return self.simp(start_imgs)
        # s = self.start_state_encoder(start_imgs)
        # g = self.goal_state_encoder(goal_imgs)
        # x = torch.cat((s, g), dim=-1)
        # x = self.action_predictor(x)
        # return x


class PPOActor(PPOBaseModel):
    def __init__(self, action_size: int):
        super(PPOActor, self).__init__()
        self.out_action = nn.Sequential(
            nn.Linear(192, 192),
            nn.ReLU(True),
            nn.Linear(192, action_size),
            nn.Softmax(dim=-1),
        )
        self.out_robot_pose = nn.Sequential(
            nn.Linear(192, 192), nn.ReLU(True), nn.Linear(192, 3)
        )
        self.out_goal_pose = nn.Sequential(
            nn.Linear(192, 192), nn.ReLU(True), nn.Linear(192, 2)
        )

    def forward(self, imgs: torch.Tensor):
        x = super().forward(imgs)
        a = self.out_action(x)
        r = self.out_robot_pose(x)
        g = self.out_goal_pose(x)
        return a, r, g


class PPOCritic(PPOBaseModel):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(193, 192), nn.ReLU(), nn.Linear(192, 1))

    def forward(self, imgs: torch.Tensor, T: torch.Tensor):
        x = super().forward(imgs)
        x = torch.cat((x, T), dim=-1)
        x = self.out(x)
        return x


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, embedding_size: int = 1024, action_size: int = 7):
        super(ConvolutionalAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56x56
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.ReLU(True),
            nn.Conv2d(
                256, embedding_size, kernel_size=7, stride=1, padding=0
            ),  # 8x8 (output size can be adjusted by kernel size)
        )

        self.action_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * embedding_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1),
        )
        self.out_robot_pose = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * embedding_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 3),
        )
        self.out_goal_pose = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * embedding_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                embedding_size, 256, kernel_size=7, stride=1, padding=0
            ),  # 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 28x28
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 56x56
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 112x112
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 224x224
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        robot_pose = self.out_robot_pose(x)
        goal_pose = self.out_goal_pose(x)
        action = self.action_predictor(x)
        reconstructed = self.decoder(x)
        return action, reconstructed, robot_pose, goal_pose
