"""LLM bandit agent with exponential temperature decay over the horizon."""

import numpy as np

from .llm import LLMBanditAgent


class ScheduledTempLLMAgent:
    """
    Wraps LLMBanditAgent with an exponential temperature schedule.
    Temperature decays from temp_start to temp_end over the horizon.
    """

    def __init__(
        self,
        horizon=200,
        temp_start=2.0,
        temp_end=0.0,
        schedule_k=5,
        max_new_tokens=200,
    ):
        self.horizon = horizon
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)
        self.schedule_k = schedule_k
        self.max_new_tokens = int(max_new_tokens)
        self.t = 0
        self._llm_agent = LLMBanditAgent()

    def current_temperature(self):
        k = self.schedule_k
        frac = self.t / self.horizon
        growth = (np.exp(k * frac) - 1) / (np.exp(k) - 1)
        return float(self.temp_start + growth * (self.temp_end - self.temp_start))

    def act(self, system_prompt, user_prompt, max_new_tokens=None):
        temp = float(self.current_temperature())
        ntok = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        action, raw_response = self._llm_agent.act(
            system_prompt, user_prompt, temperature=temp, max_new_tokens=ntok
        )
        self.t += 1
        return action, raw_response
