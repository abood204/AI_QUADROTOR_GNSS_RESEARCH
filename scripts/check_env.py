from src.environments.airsim_env import AirSimDroneEnv
from stable_baselines3.common.env_checker import check_env

def main():
    print("Checking AirSimDroneEnv...")
    # Mock IP not needed if loopback, but let's assume default
    env = AirSimDroneEnv()
    
    # Check Gymnasium compliance
    # We might need to handle the 'NotConnected' error if AirSim isn't running
    # This script assumes AirSim IS running.
    print("Resetting Env...")
    env.reset()
    
    print("Running check_env...")
    check_env(env)
    print("Environment is compliant!")
    
    print("Taking a few steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Reward={reward:.3f} Term={terminated}")
        if terminated:
            env.reset()
            
    env.close()

if __name__ == "__main__":
    main()
