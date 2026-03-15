import { Command } from "./Command.js";
export class RedXMoveCommand extends Command {
    constructor(destination) {
        super();
        this.destination = destination;
    }
    clone() {
        return new RedXMoveCommand(this.destination.clone());
    }
}
