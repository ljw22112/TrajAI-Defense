import torch
import torch.nn as nn
import torch.optim as optim

class PINNMissilePredictor(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=3, num_layers=2):
        super(PINNMissilePredictor, self).__init__()
        
        # 시계열 인코더 (Bi-LSTM)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # 궤도 및 탄착지 추론 헤드
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim) # [x_future, y_future, z_future]
        )

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        pred_pos = self.fc(last_hidden)
        return pred_pos

class PINNLoss(nn.Module):
    def __init__(self, lambda_phys=0.1, g=9.81):
        super(PINNLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_phys = lambda_phys
        self.g = g

    def forward(self, pred_pos, target_pos, last_velocity, dt=0.05):
        # 1. Data Loss: 실제 위치 오차
        data_loss = self.mse(pred_pos, target_pos)

        # 2. Physics Loss: 물리 법칙 (중력 가속도 및 등가속도 운동 법칙 준수)
        expected_z_step = last_velocity[:, 2] * dt - 0.5 * self.g * (dt**2)
        phys_loss_z = torch.mean((pred_pos[:, 2] - expected_z_step)**2)

        # Total Loss = Data Loss + lambda * Physics Loss
        total_loss = data_loss + self.lambda_phys * phys_loss_z
        return total_loss, data_loss, phys_loss_z

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"실행 디바이스: {device}")

    model = PINNMissilePredictor(input_dim=9, hidden_dim=64, output_dim=3).to(device)
    criterion = PINNLoss(lambda_phys=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dummy_input = torch.randn(32, 20, 9).to(device)
    dummy_target = torch.randn(32, 3).to(device)
    last_vel = dummy_input[:, -1, 3:6]

    optimizer.zero_grad()
    pred = model(dummy_input)
    loss, d_loss, p_loss = criterion(pred, dummy_target, last_vel)
    loss.backward()
    optimizer.step()

    print(f"PINN 학습 1회 테스트 성공!")
    print(f" - Total Loss: {loss.item():.4f}")
    print(f" - Data Loss: {d_loss.item():.4f}")
    print(f" - Physics Loss: {p_loss.item():.4f}")
