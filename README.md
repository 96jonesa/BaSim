# BaSim (modified by 96jonesa)

Fork of Henke's Old School RuneScape Barbarian Assault simulator. At the time of forking, the simulator only supported Runners, Defenders, and Collectors.

I intend to expand the simulator to include all relevant components required to run reinforcement learning on it for the purpose of solving for best strategies. Before implementing the entire game, I will polish the existing code, add Penance Healers and Player Healer, and give reinforcement learning a whirl on solving for useful Defender+Healer strategies in absence of Attackers.

I am using this as an excuse to learn JavaScript + Node.js and test out reinforcement learning on JavaScript. (This seems a more worthwhile and less excruciating approach than reimplementing in C++).

# Features Added

- Different foods
- Call changes
- Trap decay (toggleable)
- Trap repair
- Log pickup
- Hammer pickup
- Customizable tick duration
- Pause/unpause
- Save/load game state
- Toggleable infinite food
- Toggleable hammer+log requirement to repair
