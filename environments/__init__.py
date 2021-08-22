from gym.envs.registration import register as _register_env

from .daytrading_env import DayTradingEnv

_register_env(
    id='DayTrading-v0',
    entry_point='environments:DayTradingEnv'
)
