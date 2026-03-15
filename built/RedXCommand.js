import { Command } from "./Command.js";
export class RedXCommand extends Command {
    constructor(healerId) {
        super();
        this.healerId = healerId;
    }
    clone() {
        return new RedXCommand(this.healerId);
    }
}
