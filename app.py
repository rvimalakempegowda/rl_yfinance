import streamlit as st
import gymnasium as gym
import gym_anytrading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load GME trading data
df = pd.read_csv('AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Streamlit app
st.title('Stock Trading RL App')

# Display loaded data
st.write('Loaded Data:')
st.write(df.head())

# Create the environment
env = gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)

# Display environment features and prices
st.write('Environment Features:')
st.write(env.unwrapped.signal_features)
st.write('Environment Prices:')
st.write(env.unwrapped.prices)

# Explore the environment
st.write('Environment Exploration:')
num_steps = st.slider('Select the number of steps to explore:', min_value=1, max_value=100, value=10)
st.write(f'Exploring environment for {num_steps} steps...')

observation = env.reset()
for step in range(num_steps):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    st.write(f"Step {step + 1}: Action - {action}, Reward - {reward}, Done - {done}")

    if done:
        st.write("Exploration Complete.")
        break
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.cla()
env.unwrapped.render_all()
st.pyplot()


# Load the pre-trained A2C model
model = A2C.load("A2C.h5")

# Test the loaded model

st.write('Testing Loaded Model...')
test_env = gym.make('stocks-v0', df=df, frame_bound=(5, 110), window_size=5)
obs, info = test_env.reset()

while True:
    action, _states = model.predict(obs)
    observation, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

    if done:
        st.write("Test Complete.")
        st.write('Info:')
        st.write(pd.DataFrame.from_dict(info, orient='index', columns=['Value']))
        break

# Plot the results
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write('Rendering Results...')
plt.figure(figsize=(15, 6))
plt.cla()
test_env.render_all()
st.pyplot()
