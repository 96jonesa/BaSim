import { Position } from "./Position.js";
import { Food } from "./Food.js";
import { Player } from "./Player.js";
/**
 * Represents a Barbarian Assault defender player.
 */
export class DefenderPlayer extends Player {
    constructor(position) {
        super(position);
        this.foodBeingPickedUp = null;
        this.isPickingUpLogs = false;
        this.repairTicksRemaining = 0;
        this.pathQueueIndex = 0;
        this.pathQueuePositions = [];
        this.shortestDistances = [];
        this.waypoints = [];
        this.ticksStandingStill = 0;
        this.logsInInventory = 0;
        this.foodInInventory = {
            TOFU: 9,
            CRACKERS: 9,
            WORMS: 9,
        };
    }
    /**
     * @inheritDoc
     */
    tick(barbarianAssault) {
        this.ticksStandingStill++;
        if (this.repairTicksRemaining > 0) {
            this.repair(barbarianAssault);
        }
        if (this.foodBeingPickedUp !== null) {
            this.pickUpFood(barbarianAssault);
        }
        if (this.isPickingUpLogs) {
            this.pickUpLogs(barbarianAssault);
        }
        const position = this.position.clone();
        this.move();
        if (!this.position.equals(position)) {
            this.ticksStandingStill = 0;
        }
    }
    /**
     * Drops a {@link Food} of the given FoodType at this defender player's current
     * position, in the given BarbarianAssault game.
     *
     * @param barbarianAssault  the BarbarianAssault game for the food to be dropped in
     * @param foodType          the FoodType of the Food to be dropped
     */
    dropFood(barbarianAssault, foodType) {
        if ((this.foodInInventory[foodType] > 0 || barbarianAssault.infiniteFood) && this.repairTicksRemaining === 0) {
            this.foodInInventory[foodType]--;
            barbarianAssault.map.addFood(new Food(new Position(this.position.x, this.position.y), foodType, foodType === barbarianAssault.defenderFoodCall));
        }
    }
    /**
     * If this defender player is in range to repair a damaged trap in the given BarbarianAssault
     * game, then starts repairing.
     *
     * @param barbarianAssault  the BarbarianAssault game to check for damaged traps that this
     *                          defender player is in range to repair
     */
    startRepairing(barbarianAssault) {
        if (this.repairTicksRemaining === 0 && ((this.isInRepairRange(barbarianAssault.eastTrapPosition) && barbarianAssault.eastTrapCharges < 2) || (this.isInRepairRange(barbarianAssault.westTrapPosition) && barbarianAssault.westTrapCharges < 2))) {
            if (this.logsInInventory > 0 || !barbarianAssault.requireLogs) {
                this.repairTicksRemaining = (this.ticksStandingStill === 0) ? 6 : 5;
            }
        }
    }
    /**
     * Repairs a trap that is in the repair range of this defender player (if both the east and
     * west traps are in the repair range, then repairs only the east trap) in the given
     * BarbarianAssault game. This sets the trap back to two charges and removes a single log
     * from this defender player's inventory, if a trap was repaired. Then, regardless of whether
     * a trap was repaired, places this defender player in a post-repair state (no destination,
     * not picking up food, not picking up logs, and not repairing).
     *
     * @param barbarianAssault
     * @private
     */
    repair(barbarianAssault) {
        if (this.repairTicksRemaining === 1) {
            this.repairTrap(barbarianAssault);
        }
        this.repairTicksRemaining--;
        this.pathQueueIndex = 0;
        this.foodBeingPickedUp = null;
        this.isPickingUpLogs = false;
    }
    /**
     * Repairs a trap that is in the repair range of this defender player (if both the east and
     * west traps are in the repair range, then repairs only the east trap) in the given
     * BarbarianAssault game. This sets the trap back to two charges and removes a single log
     * from this defender player's inventory, if a trap was repaired.
     *
     * @param barbarianAssault  the BarbarianAssault game to repair an in-range trap in
     * @private
     */
    repairTrap(barbarianAssault) {
        if (this.isInRepairRange(barbarianAssault.eastTrapPosition)) {
            barbarianAssault.eastTrapCharges = 2;
            this.logsInInventory--;
        }
        else if (this.isInRepairRange(barbarianAssault.westTrapPosition)) {
            barbarianAssault.westTrapCharges = 2;
            this.logsInInventory--;
        }
    }
    /**
     * Picks up a {@link Food} at the same position as this defender player, with {@link FoodType}
     * equal to the type of food this defender player is picking up. Then, regardless of whether
     * any food was picked up, places this defender in a post-pickup state (no destination,
     * not picking up food, and not picking up logs).
     *
     * @param barbarianAssault  the BarbarianAssault game to pick up food in
     * @private
     */
    pickUpFood(barbarianAssault) {
        if (this.foodBeingPickedUp === null) {
            return;
        }
        const foodZone = barbarianAssault.map.getFoodZone(this.position.x >>> 3, this.position.y >>> 3);
        for (let i = 0; i < foodZone.foodList.length; i++) {
            const food = foodZone.foodList[i];
            if (this.position.x === food.position.x && this.position.y === food.position.y && food.type === this.foodBeingPickedUp) {
                foodZone.foodList.splice(i, 1);
                this.foodInInventory[this.foodBeingPickedUp]++;
                break;
            }
        }
        this.pathQueueIndex = 0;
        this.foodBeingPickedUp = null;
        this.isPickingUpLogs = false;
    }
    /**
     * Picks up a log at the same position as this defender player. Then, regardless of whether
     * a log was picked up, places this defender in a post-pickup state (no destination,
     * not picking up food, and not picking up logs).
     *
     * @param barbarianAssault  the BarbarianAssault game to pick up food in
     * @private
     */
    pickUpLogs(barbarianAssault) {
        if (!this.isPickingUpLogs) {
            return;
        }
        if (this.position.equals(barbarianAssault.northwestLogsPosition)) {
            if (barbarianAssault.northwestLogsArePresent) {
                this.logsInInventory++;
                barbarianAssault.northwestLogsArePresent = false;
            }
        }
        else if (this.position.equals(barbarianAssault.southeastLogsPosition)) {
            if (barbarianAssault.southeastLogsArePresent) {
                this.logsInInventory++;
                barbarianAssault.southeastLogsArePresent = false;
            }
        }
        this.pathQueueIndex = 0;
        this.foodBeingPickedUp = null;
        this.isPickingUpLogs = false;
    }
    /**
     * Determines if this defender player is in range to repair a trap at the given position.
     *
     * @param position  the position to check if this defender player is in range to repair
     *                  a trap at
     * @return          true if this defender player is in range to repair a trap at the
     *                  given position, otherwise false
     * @private
     */
    isInRepairRange(position) {
        return Math.abs(this.position.x - position.x) + Math.abs(this.position.y - position.y) <= 1;
    }
    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    clone() {
        let defenderPlayer = new DefenderPlayer(this.position);
        defenderPlayer.position = this.position === null ? null : this.position.clone();
        defenderPlayer.foodBeingPickedUp = this.foodBeingPickedUp;
        defenderPlayer.isPickingUpLogs = this.isPickingUpLogs;
        defenderPlayer.repairTicksRemaining = this.repairTicksRemaining;
        defenderPlayer.pathQueueIndex = this.pathQueueIndex;
        defenderPlayer.pathQueuePositions = [];
        for (let i = 0; i < this.pathQueuePositions.length; i++) {
            defenderPlayer.pathQueuePositions.push(this.pathQueuePositions[i] === null ? null : this.pathQueuePositions[i].clone());
        }
        defenderPlayer.shortestDistances = [...this.shortestDistances];
        defenderPlayer.waypoints = [...this.waypoints];
        defenderPlayer.ticksStandingStill = this.ticksStandingStill;
        defenderPlayer.logsInInventory = this.logsInInventory;
        defenderPlayer.foodInInventory = Object.assign({}, this.foodInInventory);
        return defenderPlayer;
    }
}
