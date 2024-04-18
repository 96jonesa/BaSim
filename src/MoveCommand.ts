import {Command} from "./Command.js";
import {Position} from "./Position.js";

export class MoveCommand extends Command {
    public destination: Position;

    public constructor(destination: Position) {
        super();

        this.destination = destination;
    }

    public clone(): MoveCommand {
        return new MoveCommand(this.destination);
    }
}