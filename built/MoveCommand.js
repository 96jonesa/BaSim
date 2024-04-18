import { Command } from "./Command.js";
export class MoveCommand extends Command {
    constructor(destination) {
        super();
        this.destination = destination;
    }
    clone() {
        return new MoveCommand(this.destination);
    }
}
