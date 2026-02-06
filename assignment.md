# Reinforcement Learning – Final Project (2026)

## Project Overview
The main goal of this project is to summarize the key topics (theoretical & practical) covered in the second part of the course, integrating theoretical and practical aspects.

This project focuses on solving **MiniGrid** environments using **Deep Reinforcement Learning** techniques learned in the course.

MiniGrid documentation:  
https://minigrid.farama.org/environments/minigrid/

---

## MiniGrid Environments

### 1. SimpleGridEnv
- **Description:** An 8×8 empty room  
- **Goal:** Navigate to the green goal square  
- **Action Space:**  
  - Turn Left  
  - Turn Right  
  - Move Forward  

### 2. KeyDoorBallEnv
- **Description:** An 8×8 grid divided into two rooms  
- **Mission Sequence:**  
  1. Find the Key (left room)  
  2. Open the Door  
  3. Pick up the Ball (right room)  
  4. Reach the Goal  
- **Action Space:**  
  - Turn Left  
  - Turn Right  
  - Move Forward  
  - Pickup  
  - Toggle  

> In both environments, reward shaping is allowed.  
> Unlike the mid‑semester project, **observations are pixel‑based images**.

---

## Notebook Template
Use the following template notebook as the foundation for your project.  
You must save a copy and implement your solution directly inside it:

https://colab.research.google.com/drive/17jaQ11Ysl6PYp8tS8-UFmIUR73qapuHZ?usp=sharing

The template already includes:
- Environment setup
- Random‑action demonstrations

---

## Project Requirements

You should:
- Demonstrate **Deep RL knowledge**
- Try to solve the environments in **as few episodes as possible** (competition element)
- Use **multiple Deep RL algorithms** taught in the course
- Discuss **advantages and disadvantages** of each approach
- Compare methods using **relevant graphs**
- Apply **image preprocessing**
- Experiment with **hyperparameters**, including:
  - Learning rate
  - Epsilon
  - Replay buffer size
  - Target network update rate
  - Initialization strategies
- Address **exploration vs exploitation**
- Use different considerations for each environment

### Evaluation Metrics
- Training graphs (reward / steps vs episodes)
- Inference stage:
  - Average number of steps over the **last 100 episodes**
- Clear discussion of **strengths and weaknesses** of each approach

---

## Guidelines

1. **Do not use existing RL libraries** – implement algorithms yourself  
2. **Only use algorithms taught in the course**  
3. For each experiment, show:
   - Number of steps to solve the environment
   - Convergence graphs (reward/steps vs episodes)
4. Include **video clips**:
   - During training
   - After convergence
5. If an environment is not fully solved:
   - Report average rewards over 100 episodes
6. Submit:
   - Final report (PDF)
   - Google Colab notebook link
7. Write **clean, well‑structured code**
8. **Do not modify notebooks after submission**
9. The final report must be:
   - Professional
   - Self‑contained
   - Understandable without referencing the code

---

## Submission Details

- **Report filename:** `report_ID1_ID2.pdf`
- **Max pages:** 10
- Include:
  - Project title
  - Names
  - Student IDs

You may also submit an optional **explainer file** with usage instructions.

### Required Submission Files
1. `details.txt`  
2. `report_ID1_ID2.pdf`  
3. `explainer.txt` (optional)

### `details.txt` format example:
```
Link to the notebook:
https://colab.research.google.com/...

Full name student number 1:
ID student number 1:

Full name student number 2:
ID student number 2:
```

---

## Team Size
- Groups of **2 students**

---

## Academic Integrity

- Copying code or answers is **strictly forbidden**
- Sharing code with other students is also a violation
- Any reused code must be:
  - Properly cited
  - Clearly marked with your contribution
- Violations will be reported and handled according to university policy
