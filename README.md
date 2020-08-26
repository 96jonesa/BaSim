# BaSim (modified and expanded by 96jonesa)

Fork of Henke's Old School RuneScape Barbarian Assault simulator. At the time of forking, the simulator only supported Runners, Defenders, and Collectors.

I intend to expand the simulator to include all relevant components required to run reinforcement learning on it for the purpose of solving for best strategies. Before implementing the entire game, I will polish the existing code, add Penance Healers and Player Healer, and give reinforcement learning a whirl on solving for useful Defender+Healer strategies in absence of Attackers. The game is very large (hundreds of state space dimensions, hundreds of actions per state, and ~100 ticks in duration), so configuring the learning process and deciding on appropriate abstractions will be no small feat.

I am using this as an opportunity to learn JavaScript + Node.js and test out reinforcement learning on JavaScript. (This seems a more worthwhile approach than implementing in C++). I have no prior experience writing JavaScript - this is very much a project in learning JavaScript. As a result, the structure/organization of the code will evolve as my awareness of and need for it expand. 

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
