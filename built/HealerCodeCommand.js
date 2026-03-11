import { Command } from "./Command.js";
export class HealerCodeCommand extends Command {
    constructor(healerId, count) {
        super();
        this.healerId = healerId;
        this.count = count;
    }
    clone() {
        return new HealerCodeCommand(this.healerId, this.count);
    }
}
