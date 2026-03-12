import { Command } from "./Command.js";
export class WalkRunCommand extends Command {
    constructor(isRunning) {
        super();
        this.isRunning = isRunning;
    }
    clone() {
        return new WalkRunCommand(this.isRunning);
    }
}
