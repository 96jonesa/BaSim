import { Command } from "./Command.js";
export class DefenderActionCommand extends Command {
    constructor(type) {
        super();
        this.type = type;
    }
    clone() {
        return new DefenderActionCommand(this.type);
    }
}
