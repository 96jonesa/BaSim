import { Command } from "./Command.js";
export class DefenderPickupAtCommand extends Command {
    constructor(position) {
        super();
        this.position = position;
    }
    clone() {
        return new DefenderPickupAtCommand(this.position.clone());
    }
}
