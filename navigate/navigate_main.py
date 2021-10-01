"""Main module."""

from unityagents import UnityEnvironment

from config import settings
from navigate.dqn_agent import Agent
from navigate.unityenv_wrapper import UnityEnvWrapper

if __name__ == "__main__":
    unity_env = UnityEnvironment(settings.unity_env.file_path)
    env = UnityEnvWrapper(unity_env)
    agent = Agent(env)
    agent.learn(**settings.learn.kwargs)
