import numpy as np
import pandas as pd

class MissilePhysicsSimulator:
    def __init__(self, g=9.81, rho0=1.225, H=8500):
        self.g = g          # 중력 가속도 (m/s^2)
        self.rho0 = rho0    # 해수면 대기 밀도 (kg/m^3)
        self.H = H          # 대기 스케일 고도 (m)

    def get_air_density(self, z):
        """고도 z(m)에 따른 대기 밀도 계산 (US Standard Atmosphere)"""
        return self.rho0 * np.exp(-np.maximum(0, z) / self.H)

    def simulate_trajectory(self, v0, angle_deg, mass_init, cd, area, burn_time, thrust, dt=0.05):
        """6-DOF 비행체 운동 방정식 수치 해석 (Euler Method)"""
        angle_rad = np.radians(angle_deg)
        x, y, z = 0.0, 0.0, 0.0
        vx = v0 * np.cos(angle_rad)
        vy = 0.0
        vz = v0 * np.sin(angle_rad)
        
        mass = mass_init
        t = 0.0
        trajectory = []

        while z >= 0.0:
            rho = self.get_air_density(z)
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            
            # 공기저항력 F_drag = 0.5 * Cd * rho * A * v^2
            f_drag = 0.5 * cd * rho * area * (speed**2) if speed > 0 else 0
            
            # 추력 및 질량 감쇄
            current_thrust = thrust if t <= burn_time else 0.0
            mass_dot = (mass_init * 0.6) / burn_time if t <= burn_time else 0.0
            mass = np.maximum(mass_init * 0.4, mass - mass_dot * dt)

            # 가속도 계산 (F = ma => a = F/m)
            ax = (current_thrust * (vx / speed) - f_drag * (vx / speed)) / mass if speed > 0 else 0
            ay = 0.0
            az = -self.g + ((current_thrust * (vz / speed) - f_drag * (vz / speed)) / mass if speed > 0 else 0)

            trajectory.append([t, x, y, z, vx, vy, vz, ax, ay, az, mass])

            # 위치 및 속도 업데이트
            x += vx * dt
            y += vy * dt
            z += vz * dt
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            t += dt

            if t > 600: # 안전 타임아웃
                break

        return np.array(trajectory)

def generate_dataset(num_samples=500, save_path="missile_trajectories.npy"):
    sim = MissilePhysicsSimulator()
    dataset = []

    print(f"총 {num_samples}개의 물리 시뮬레이션 합성 데이터 생성 중...")
    for i in range(num_samples):
        v0 = np.random.uniform(300, 800)       # 초기 속도 (m/s)
        angle = np.random.uniform(30, 75)      # 발사 각도 (deg)
        mass = np.random.uniform(800, 1500)    # 초기 질량 (kg)
        cd = np.random.uniform(0.15, 0.35)     # 항력 계수
        area = np.random.uniform(0.2, 0.5)     # 단면적 (m^2)
        burn_time = np.random.uniform(5, 15)   # 연소 시간 (s)
        thrust = np.random.uniform(20000, 50000) # 추력 (N)

        traj = sim.simulate_trajectory(v0, angle, mass, cd, area, burn_time, thrust)
        dataset.append(traj)

    print(f"데이터 생성 완료. '{save_path}' 파일로 저장합니다.")
    np.save(save_path, np.array(dataset, dtype=object))

if __name__ == "__main__":
    generate_dataset(num_samples=500)
