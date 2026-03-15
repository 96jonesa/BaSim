import {Command} from "./Command.js";
import {Position} from "./Position.js";

export class RedXMoveCommand extends Command {
    public readonly destination: Position;

    public constructor(destination: Position) {
        super();
        this.destination = destination;
    }

    public clone(): RedXMoveCommand {
        return new RedXMoveCommand(this.destination.clone());
    }
}
