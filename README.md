# 📄 Project Proposal  
## **Multi-Agent Drone Navigation with LLM-Guided Reasoning: Exploring Non-Stationarity in Multi-Agent RL**

---

## 🎯 Motivation
- Modern RL is effective in **stationary single-agent environments**, but struggles in **non-stationary, multi-agent settings**.  
- The rise of **agentic AI** — autonomous agents reasoning and interacting with each other — is a hot direction in industry and research, including at IBM.  
- This project explores a realistic, interactive scenario: **multiple drones navigating a shared environment while reasoning about other agents.**
- Optional: Integrate **LLM-based reasoning** to complement RL policy and mitigate non-stationarity.

---

## 🧩 Goals
- ✅ Build a simulated multi-agent environment (e.g., drones navigating a warehouse).  
- ✅ Implement and compare several training strategies:
  - Independent RL agents (e.g., Q-learning, PPO) — *baseline*
  - Centralized training, decentralized execution (CTDE)
  - LLM-guided agents that reason about the intentions of others  
- ✅ Quantify the effects of **non-stationarity**.  
- ✅ Experiment with equilibria concepts (e.g., Nash equilibrium, best-response dynamics).

---

## 🔷 Environment
- **Simulator options:**
  - [Gymnasium](https://gymnasium.farama.org/) + [PettingZoo](https://pettingzoo.farama.org/) (multi-agent friendly, quick to set up)  
  - [AirSim](https://microsoft.github.io/AirSim/) or [Colosseum](https://github.com/CodexLabsLLC/Colosseum) (realistic drone physics and visuals)

- **Observation space:**  
  Drone’s own position, velocity, distance to other drones, map occupancy grid.

- **Action space:**  
  Discrete (e.g., turn left/right, move forward/stop) or continuous (e.g., thrust vector, yaw rate).

---

## 🧠 Agent Architectures

| Approach              | Description |
|-----------------------|-------------|
| **Independent RL**    | Each drone runs its own policy, ignoring others |
| **Centralized RL**    | One policy that takes all agents’ states into account |
| **LLM-guided RL**     | Each agent queries an LLM for reasoning about others |
| **Game-theoretic RL** | Agents use best-response dynamics or fictitious play to converge to equilibria |

### Example LLM prompt:
 'Given my position X and the positions of drones Y1, Y2, Y3, what should I do to minimize collisions and reach the goal?'

---

## 📊 Metrics to Evaluate
- Average time to reach goal  
- Collision rate  
- Policy stability over training episodes  
- Reward evolution over time

---

## 🛠️ Why This Aligns with IBM Strategic Directions
- ✅ **Agentic AI:** multi-agent autonomy focus  
- ✅ **RL for reasoning:** augment RL with LLM reasoning  
- ✅ **Fundamental RL research:** stability in non-stationary environments  

---

## 🪜 Next Steps
1. Choose a simulator: Gym (fast) vs AirSim/Colosseum (realistic).  
2. Define the multi-agent task and specify observation/action spaces.  
3. Build baseline RL agents and environment.  
4. Implement LLM-guided policy integration.  
5. Run experiments comparing all approaches.  
6. Analyze convergence, stability, and equilibria.

---

## 🔗 Optional: Deliverables
- 📄 1-page formal proposal document (optional)  
- 🧰 Suggested libraries and setup commands  
- 📝 Example training loop outline in code  
- 📚 Survey of related papers

---
